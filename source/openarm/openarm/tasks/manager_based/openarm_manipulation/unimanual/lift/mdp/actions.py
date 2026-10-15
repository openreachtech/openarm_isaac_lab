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

"""A gripper action that can squeeze, not just open and shut."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MimicJointPositionAction(ActionTerm):
    """One continuous action driving several joints to the same position target.

    Replaces ``BinaryJointPositionAction`` for the gripper. That term can only pick between
    two constants, which means the grip force is not something the policy chooses: the
    fingers run on an implicit actuator, so the force they apply is
    ``stiffness * (target - actual_position)``, and with the target pinned at 0.0 that comes
    out as ``stiffness * half_the_object_width``. A narrow object is gripped weakly and a
    wide one firmly, whatever the policy would prefer.

    That was measured to be the dominant failure mode on the old cube task: across every
    shape the approach and the initial grasp succeeded 100% of the time, and the rate of
    losing the object afterwards tracked the grasp width monotonically (3.8 cm -> 3.1%,
    2.2 cm -> 25.8%). Moving the close command to -0.01 lifted the overall success rate from
    90.2% to 92.1% with no retraining at all.

    So the target is made continuous here, which also gives the privileged ``object_mass``
    and ``object_friction`` something to act on -- without it the policy is told how slippery
    the object is and has no way to respond.

    Isaac Lab has no stock term for this. ``JointPositionActionCfg`` over two finger joints
    would expose two independent actions; only ``BinaryJointAction`` does the one-to-many
    broadcast, and only for two fixed values. Independent fingers are interesting -- the RMA
    paper's EGAD example picks a flat object up by pressing with one finger and sliding the
    other underneath -- but they would confound the question of whether grip force itself is
    what was missing, so the two are tied together here.
    """

    cfg: MimicJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: MimicJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # The target is an affine map of the action rather than a fraction of the joint
        # range, because the useful range runs *past* the closed limit: driving the target
        # below zero is what produces grip force once the fingers are already touching.
        lo, hi = self.cfg.target_range
        self._offset = 0.5 * (hi + lo)
        self._scale = 0.5 * (hi - lo)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        target = self._offset + self._scale * actions.clamp(-1.0, 1.0)
        self._processed_actions[:] = target.expand(-1, self._num_joints)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class MimicJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`MimicJointPositionAction`."""

    class_type: type[ActionTerm] = MimicJointPositionAction

    joint_names: list[str] = None
    """Joint name expressions. All matched joints receive the same target."""

    target_range: tuple[float, float] = (-0.020, 0.044)
    """Joint position the action maps onto, for ``action = -1`` and ``action = +1``.

    0.044 is the finger's open limit. The lower end is deliberately past the closed limit:
    with ``stiffness = 2e3`` per finger, a target of -0.020 against an object of half-width
    2.2 cm asks for about 84 N, where the old fixed target of 0.0 asked for 44 N.

    It stops at -0.020 rather than going further because -0.04 was measured to hurt: overall
    lift held at 91.8% but placement within 5 cm fell from 84.2% to 78.3%, the object being
    squeezed out of position as the fingers closed. -0.01 and -0.02 both behaved well.
    """
