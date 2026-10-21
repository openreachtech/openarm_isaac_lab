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

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...lift.mdp.events import arm_init_levels
from ...lift.mdp.rewards import lift_threshold
from .rewards import object_in_bin, place_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_init_level_place(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    margin: float = 0.02,
) -> float:
    """``arm_init_level`` with placement as the bar.

    The lift version cannot be reused. It counts the object as held when its centre is
    above an absolute 4 cm, and a sphere resting in the bin is at 2.5 cm, so every success
    would be read as a drop and demoted. And it promotes on ending within 10 cm of the goal,
    which here would include leaving the sphere beside the bin.

    Promote on success; hold while the object is lifted or in the bin, placed but not yet
    let go of; demote otherwise.
    """
    levels = arm_init_levels(env)
    obj = env.scene["object"]
    success = place_success(env)[env_ids]
    lifted = obj.data.root_pos_w[env_ids, 2] > lift_threshold(env, margin)[env_ids]
    held = lifted | object_in_bin(env)[env_ids]

    step = torch.where(success, 1, torch.where(held, 0, -1))
    num_levels = env.cfg.events.reset_robot_joints.params["num_levels"]
    levels[env_ids] = torch.clamp(levels[env_ids] + step, 0, num_levels)
    return float(levels.float().mean())


def episode_place_success(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
    """Fraction of the episodes ending now that ended in success. Changes nothing.

    A curriculum term only because that is where Isaac Lab logs a per-reset scalar
    (``Curriculum/<name>``); the reward sums say how much was earned, not how often the
    task was finished.
    """
    return float(place_success(env)[env_ids].float().mean())
