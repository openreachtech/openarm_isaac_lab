# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Whether the object was put down by hand or dropped -- decided at the moment of release.

Arm-Place-Sphere-Box asks for the sphere at the box's centre, and on its own that did not
stop the policy dropping it from above the rim: at model_1000, 90% of episodes ended with a
sphere dropped from over 6 cm (58% from over 10 cm) and resting off the mark, and the hand was
inside the box at only 11% of releases. So the box rewards are made to depend on *how* the
sphere arrived: released with its centre within ``PLACE_RELEASE_HEIGHT`` of where it rests
on the floor. With 8 cm walls that means the hand went down into the box.

This needs memory of the release itself, which no single frame holds, so the state lives on
the env (as the arm-init levels and the object catalog do) and is advanced once per step by
whichever term reads it first.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .rewards import is_grasped, place_target_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

RELEASE_STEPS = 3
"""A release is a grasp held this many steps followed by this many steps without one.

Finger contact can drop out for a step mid-carry; counting that as a release would judge the
sphere on a moment when it was never let go of.
"""

PLACE_RELEASE_HEIGHT = 0.02
"""How far above its resting height the object's centre may be when let go of. A sphere
released from here falls at most 2 cm; from above an 8 cm rim, at least 8."""

RELEASE_TRACKER_ATTR = "release_tracker"


class ReleaseTracker:
    """Per-environment release state.

    ``placed``       the last release was low enough to count as placing (cleared on re-grasp)
    ``last_height``  object centre above its resting height at the last release, NaN if none
    """

    def __init__(self, env: ManagerBasedEnv) -> None:
        n, device = env.num_envs, env.device
        self.grasp_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.free_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.armed = torch.zeros(n, dtype=torch.bool, device=device)
        self.pending_height = torch.zeros(n, device=device)
        self.placed = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_height = torch.full((n,), float("nan"), device=device)
        self._step = -1

    def update(self, env: ManagerBasedRLEnv) -> None:
        """Advance by one env step. Idempotent within a step, so call order does not matter."""
        if env.common_step_counter == self._step:
            return
        self._step = env.common_step_counter

        grasped = is_grasped(env)
        self.grasp_streak = torch.where(grasped, self.grasp_streak + 1, torch.zeros_like(self.grasp_streak))
        regrasped = self.grasp_streak >= RELEASE_STEPS
        self.armed |= regrasped
        self.placed &= ~regrasped

        letting_go = self.armed & ~grasped & (self.free_streak == 0)
        height = env.scene["object"].data.root_pos_w[:, 2] - place_target_w(env)[:, 2]
        self.pending_height = torch.where(letting_go, height, self.pending_height)

        self.free_streak = torch.where(
            self.armed & ~grasped, self.free_streak + 1, torch.zeros_like(self.free_streak)
        )
        committed = self.free_streak == RELEASE_STEPS
        self.placed = torch.where(committed, self.pending_height <= PLACE_RELEASE_HEIGHT, self.placed)
        self.last_height = torch.where(committed, self.pending_height, self.last_height)
        self.armed &= ~committed

    @property
    def releasing_low(self) -> torch.Tensor:
        """Let go of low enough, but not yet confirmed as a release (fewer than RELEASE_STEPS
        steps without a grasp)."""
        return self.armed & (self.free_streak > 0) & (self.pending_height <= PLACE_RELEASE_HEIGHT)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.grasp_streak[env_ids] = 0
        self.free_streak[env_ids] = 0
        self.armed[env_ids] = False
        self.placed[env_ids] = False
        self.last_height[env_ids] = float("nan")


def release_tracker(env: ManagerBasedEnv) -> ReleaseTracker:
    tracker = getattr(env, RELEASE_TRACKER_ATTR, None)
    if tracker is None:
        tracker = ReleaseTracker(env)
        setattr(env, RELEASE_TRACKER_ATTR, tracker)
    return tracker


def placement_valid(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Still holding the object, or let go of it low enough to count as placing.

    A release still being confirmed counts too, judged by the height it began at. Without
    that, opening the fingers switched the holding rewards off at once and paid nothing for
    RELEASE_STEPS steps, and the policy learned never to let go: at model_1999 (2026-10-22,
    20 cm box) 84% of episodes ended gripping the sphere on the box floor, half of them
    within the 1 cm that letting go would have scored. A sphere dropped from above is still
    refused, by the height at which it was let go of.

    Always true unless ``env.cfg.require_hand_placement`` is set, so the shallow and deep
    tasks are unaffected.
    """
    if not getattr(env.cfg, "require_hand_placement", False):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    tracker = release_tracker(env)
    tracker.update(env)
    return is_grasped(env) | tracker.placed | tracker.releasing_low


def reset_release_tracker(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Reset event. The curriculum has already read the episode's final state by now:
    ``_reset_idx`` runs curriculum terms before reset events."""
    release_tracker(env).reset(env_ids)
