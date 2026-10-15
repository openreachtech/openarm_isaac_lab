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

"""RMA adapter + frozen RMA teacher, wired into rsl-rl's distillation runner."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules import StudentTeacherRecurrent

from openarm.assets.models.rma_actor import RmaActor
from openarm.assets.models.rma_student import RmaDepthStudent


class RmaStudentTeacher(StudentTeacherRecurrent):
    """The adapter learns ``z``; the policy that consumes it is the teacher's, frozen.

    Observation layout, fixed by the term order in ``ArmLiftStudentObservationsCfg``:

        policy  = [ x_t (state_dim) | flattened depth ]
        teacher = [ x_t (state_dim) | privileged ]
        aux     = the clean object position

    The first ``state_dim`` entries of the two groups are the same terms in the same
    order, which is what lets the teacher's policy half be applied to the student's own
    observation.

    The teacher is frozen explicitly. rsl-rl only calls ``.eval()`` on it, which is enough
    for the stock setup because ``evaluate()`` runs under ``no_grad``; here the teacher's
    policy sits *inside* the gradient path (it is how ``z_hat`` reaches the action loss) and
    the optimiser is built over ``policy.parameters()``, so without ``requires_grad_(False)``
    behaviour cloning would quietly fine-tune the teacher into agreeing with the student.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        state_dim: int,
        image_height: int,
        image_width: int,
        privileged_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dims: list[int] = [128, 64],
        category_slice: tuple[int, int] = (5, 10),
        category_embed_dim: int = 32,
        aux_dim: int = 3,
        cnn_channels: list[int] = [16, 32, 64],
        cnn_head_dims: list[int] = [128],
        student_head_dims: list[int] = [256, 128],
        student_activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        # The parent builds a placeholder MLP student, an MLP teacher and a GRU over the
        # whole observation (depth included). All three are replaced below; they cost one
        # allocation at startup and nothing after.
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        self.state_dim = int(state_dim)
        self.privileged_dim = int(privileged_dim)

        num_student_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        expected = self.state_dim + int(image_height) * int(image_width)
        if num_student_obs != expected:
            raise ValueError(
                f"Student observation is {num_student_obs} wide, expected state + H*W = "
                f"{self.state_dim} + {image_height}*{image_width} = {expected}."
            )
        num_teacher_obs = sum(obs[g].shape[-1] for g in obs_groups["teacher"])
        if num_teacher_obs != self.state_dim + self.privileged_dim:
            raise ValueError(
                f"Teacher observation is {num_teacher_obs} wide, expected state + privileged"
                f" = {self.state_dim} + {self.privileged_dim}."
            )

        self.teacher = RmaActor(
            num_policy_obs=self.state_dim,
            num_privileged_obs=self.privileged_dim,
            num_actions=num_actions,
            category_slice=category_slice,
            category_embed_dim=category_embed_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            policy_hidden_dims=kwargs.get("teacher_hidden_dims", [256, 128, 64]),
            activation=kwargs.get("activation", "elu"),
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.student = RmaDepthStudent(
            state_dim=self.state_dim,
            image_height=image_height,
            image_width=image_width,
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            cnn_channels=cnn_channels,
            cnn_head_dims=cnn_head_dims,
            rnn_type=kwargs.get("rnn_type", "gru"),
            rnn_hidden_dim=kwargs.get("rnn_hidden_dim", 128),
            rnn_num_layers=kwargs.get("rnn_num_layers", 1),
            head_dims=student_head_dims,
            activation=student_activation,
        )
        if self.teacher_recurrent:
            raise ValueError("The RMA teacher is feed-forward; set teacher_recurrent = False.")
        # The parent built a GRU over the whole observation, depth included. The student owns
        # the real one, so the parent's is dropped rather than left aliased: registering the
        # same module twice writes the weights into every checkpoint twice, which is a trap
        # for anyone reading one. The three parent methods that reached for `memory_s` are
        # overridden below.
        del self._modules["memory_s"]
        print(f"RMA student: {self.student}")

    """
    Hidden state. The parent routes these through `memory_s`, which no longer exists.
    """

    def reset(self, dones: torch.Tensor | None = None, hidden_states=(None, None)) -> None:
        self.student.memory.reset(dones, hidden_states[0])

    def get_hidden_states(self):
        return self.student.memory.hidden_state, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        self.student.memory.detach_hidden_state(dones)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        # Checkpoints written while `memory_s` was still aliased carry the GRU weights under
        # both names. They are identical, so the stale copy is simply dropped.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("memory_s.")}
        return super().load_state_dict(state_dict, strict=strict)

    """
    Forward passes.
    """

    def _student_outputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(x_t, z_hat, object position estimate)``."""
        student_obs = self.student_obs_normalizer(self.get_student_obs(obs))
        latent, aux = self.student(student_obs)
        return student_obs[..., : self.state_dim], latent, aux

    def _action(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.teacher.policy(torch.cat([state, latent], dim=-1))

    def act(self, obs: TensorDict) -> torch.Tensor:
        state, latent, _ = self._student_outputs(obs)
        mean = self._action(state, latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(
        self, obs: TensorDict, return_extras: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, latent, aux = self._student_outputs(obs)
        action = self._action(state, latent)
        return (action, latent, aux) if return_extras else action

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """The teacher's action -- the behaviour-cloning target."""
        with torch.no_grad():
            return self.teacher(self.teacher_obs_normalizer(self.get_teacher_obs(obs)))

    """
    Supervision targets.
    """

    def get_latent_target(self, obs: TensorDict) -> torch.Tensor:
        """``z*`` straight out of the teacher's encoder."""
        with torch.no_grad():
            teacher_obs = self.teacher_obs_normalizer(self.get_teacher_obs(obs))
            return self.teacher.encode(teacher_obs[..., self.state_dim :])

    def get_aux_target(self, obs: TensorDict) -> torch.Tensor:
        """The clean object position, from its own observation group."""
        return torch.cat([obs[g] for g in self.obs_groups["aux"]], dim=-1)
