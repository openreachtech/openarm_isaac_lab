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

"""The goal as a command: the point on the bin floor where the object should come to rest.

The lift task's goal is a ``UniformPoseCommand`` named ``object_pose``, and the observations
(``target_object_position``, ``object_to_goal``) read it through the command manager.
Publishing the bin target under the same name points all of them at the bin unchanged.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .rewards import place_target_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BinTargetCommand(CommandTerm):
    """Recomputed from the bin's pose every step; never resampled.

    The bin is placed by a reset event, and ``ManagerBasedRLEnv._reset_idx`` applies reset
    events before it resets the command manager, so the command never lags the bin.
    """

    cfg: BinTargetCommandCfg

    def __init__(self, cfg: BinTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.object = env.scene[cfg.object_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """``(num_envs, 7)``: target position and an identity attitude, in the robot frame."""
        return self.pose_command_b

    def _update_metrics(self):
        target_w = place_target_w(self._env)
        self.metrics["position_error"] = torch.norm(target_w - self.object.data.root_pos_w, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        self._update_command()

    def _update_command(self):
        target_w = place_target_w(self._env)
        identity = torch.zeros(self.num_envs, 4, device=self.device)
        identity[:, 0] = 1.0
        pos_b, quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, target_w, identity
        )
        self.pose_command_b[:, :3] = pos_b
        self.pose_command_b[:, 3:] = quat_b


@configclass
class BinTargetCommandCfg(CommandTermCfg):
    class_type: type = BinTargetCommand

    asset_name: str = "robot"
    object_name: str = "object"
    # Longer than any episode: the command is recomputed every step, never resampled.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
