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

"""Guards against a PhysX blow-up poisoning the returns.

The cube box task, with 30% of episodes starting on the box floor, died at iteration ~185
(2026-10-23): one iteration's mean reward read -163,904, the value loss climbed through
1e33 to inf, and the action std went NaN. The smoothness penalty is quadratic in joint
velocity, so a single environment whose solver had exploded -- most likely fingers pinning
the cube against the kinematic box -- was enough. Random actions over the same resets did
not reproduce it, so it takes the learned behaviour to get there; these do not prevent it,
they stop it from reaching the critic.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def physics_blowup(
    env: ManagerBasedRLEnv,
    max_joint_vel: float,
    max_object_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Any joint or the object moving impossibly fast, or a non-finite state."""
    joint_vel = env.scene[robot_cfg.name].data.joint_vel
    obj = env.scene[object_cfg.name].data
    speed = torch.norm(obj.root_lin_vel_w, dim=1)
    finite = torch.isfinite(joint_vel).all(dim=1) & torch.isfinite(obj.root_pos_w).all(dim=1)
    return ~finite | (joint_vel.abs() > max_joint_vel).any(dim=1) | (speed > max_object_speed)


def joint_vel_l2_clamped(
    env: ManagerBasedRLEnv, max_value: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``joint_vel_l2`` with a ceiling, so the step a blow-up is caught on costs a bounded amount."""
    joint_vel = torch.nan_to_num(env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids], nan=0.0)
    return torch.sum(torch.square(joint_vel), dim=1).clamp(max=max_value)


def action_rate_l2_clamped(env: ManagerBasedRLEnv, max_value: float) -> torch.Tensor:
    """``action_rate_l2`` with a ceiling.

    The second run of the same task (misplaced starts, blow-up guards in) got to iteration
    1250 and died there, at the step where the smoothness weights jump from -1e-4 to -1e-1:
    mean reward -146, -8.3e3, -4.4e6, ... -2.5e20 over eight iterations. Actions are not
    clipped, the action std was ~3 when the weight jumped, and the penalty on their change
    is quadratic; capped, a runaway costs a bounded amount and stops being self-feeding.
    """
    am = env.action_manager
    delta = torch.nan_to_num(am.action - am.prev_action, nan=0.0)
    return torch.sum(torch.square(delta), dim=1).clamp(max=max_value)
