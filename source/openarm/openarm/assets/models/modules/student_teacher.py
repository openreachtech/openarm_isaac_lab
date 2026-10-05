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

"""Depth student + frozen privileged teacher for rsl-rl distillation."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import StudentTeacher as RslStudentTeacher

from openarm.assets.models.depth_student import DepthStudentPolicy


class DepthStudentTeacher(RslStudentTeacher):
    """Swaps rsl-rl's MLP student for :class:`DepthStudentPolicy`; keeps the MLP teacher.

    The teacher is left untouched on purpose. ``Arm-Lift-Cube``'s PPO actor is already a
    plain MLP, so the parent's ``MLP`` reproduces it exactly as long as
    ``teacher_hidden_dims`` / ``activation`` match the checkpoint, and the parent's
    ``load_state_dict`` loads it from a PPO checkpoint by rewriting the ``actor.`` keys.

    Observation layout:
        policy  obs = [state | flattened depth]   (student)
        teacher obs = the privileged vector the PPO teacher was trained on

    ``aux_obs_slice`` points at the privileged block inside the teacher observation that
    the student has to predict (``object_position``). It is the supervision target for
    the auxiliary head.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        state_dim: int,
        image_height: int,
        image_width: int,
        aux_obs_slice: tuple[int, int],
        cnn_channels: list[int] = [16, 32, 64],
        cnn_head_dims: list[int] = [128, 128],
        trunk_dims: list[int] = [256, 128, 64],
        student_activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        # The parent builds a placeholder MLP student (and the real MLP teacher); the
        # student is replaced below.
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        self.aux_start, self.aux_end = int(aux_obs_slice[0]), int(aux_obs_slice[1])
        assert self.aux_end > self.aux_start, f"Invalid aux_obs_slice: {aux_obs_slice}."

        num_student_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        expected = int(state_dim) + int(image_height) * int(image_width)
        assert num_student_obs == expected, (
            f"Student obs width ({num_student_obs}) != state + H*W "
            f"({state_dim} + {image_height}*{image_width} = {expected}). "
            "Check the policy observation group against the model cfg."
        )
        num_teacher_obs = sum(obs[group].shape[-1] for group in obs_groups["teacher"])
        assert self.aux_end <= num_teacher_obs, (
            f"aux_obs_slice {aux_obs_slice} runs past the teacher observation ({num_teacher_obs})."
        )

        self.student = DepthStudentPolicy(
            state_dim=state_dim,
            image_height=image_height,
            image_width=image_width,
            num_actions=num_actions,
            aux_dim=self.aux_end - self.aux_start,
            cnn_channels=cnn_channels,
            cnn_head_dims=cnn_head_dims,
            trunk_dims=trunk_dims,
            activation=student_activation,
        )
        print(f"Depth student: {self.student}")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """Parent ``act()`` hands over normalized student obs; keep only the action head."""
        action, _ = self.student(obs)
        self.distribution = torch.distributions.Normal(action, self._action_std(action))

    def _action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        if self.noise_std_type == "log":
            return torch.exp(self.log_std).expand_as(mean)
        raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

    def act_inference(self, obs: TensorDict, return_aux: bool = False):
        """Student forward used by the distillation update.

        Args:
            return_aux: also return the predicted object position for the auxiliary loss.
        """
        student_obs = self.student_obs_normalizer(self.get_student_obs(obs))
        action, aux = self.student(student_obs)
        return (action, aux) if return_aux else action

    def get_aux_target(self, obs: TensorDict) -> torch.Tensor:
        """Ground-truth object position, sliced out of the privileged teacher observation."""
        return self.get_teacher_obs(obs)[..., self.aux_start : self.aux_end]
