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

"""Distillation with DextrAH-G's auxiliary object-position loss."""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.algorithms import Distillation as RslDistillation

from openarm.assets.models.modules.student_teacher import DepthStudentTeacher


class Distillation(RslDistillation):
    """Behavior cloning plus an object-position regression term.

        L = ||a_student - a_teacher|| + aux_pos_loss_coef * ||x_obj_hat - x_obj||

    This is DextrAH-G eq. in section 3.3 with beta = ``aux_pos_loss_coef``. Only
    ``update()`` differs from the parent; rollout collection, storage, the optimizer and
    the multi-GPU path are inherited.

    Note on ``gradient_length``: the student here is not recurrent, so accumulating the
    loss over ``gradient_length`` steps is plain gradient accumulation (an effective
    minibatch of ``gradient_length * num_envs``), not truncated BPTT.
    """

    policy: DepthStudentTeacher

    def __init__(self, policy: DepthStudentTeacher, aux_pos_loss_coef: float = 0.1, **kwargs) -> None:
        super().__init__(policy, **kwargs)
        self.aux_pos_loss_coef = aux_pos_loss_coef

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_aux_loss = 0.0
        loss = 0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            for obs, _, privileged_actions, dones in self.storage.generator():
                actions, aux_pred = self.policy.act_inference(obs, return_aux=True)
                aux_target = self.policy.get_aux_target(obs)

                behavior_loss = self.loss_fn(actions, privileged_actions)
                aux_loss = self.loss_fn(aux_pred, aux_target)
                loss = loss + behavior_loss + self.aux_pos_loss_coef * aux_loss

                mean_behavior_loss += behavior_loss.item()
                mean_aux_loss += aux_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    loss = 0

        mean_behavior_loss /= cnt
        mean_aux_loss /= cnt
        self.storage.clear()

        return {
            "behavior": mean_behavior_loss,
            "object_position": mean_aux_loss,
            "total": mean_behavior_loss + self.aux_pos_loss_coef * mean_aux_loss,
        }


class RmaDistillation(RslDistillation):
    """Behaviour cloning, plus regression onto the teacher's latent and the object position.

        L = ||a_s - a_t||  +  w_z ||z_hat - z*||  +  w_p ||x_obj_hat - x_obj||

    Behaviour cloning is kept as the main term rather than following Liang et al. eq. (3),
    which minimises the latent error alone. The two are not equivalent in practice: an error
    in ``z`` matters only insofar as it changes the action, and the 16 dimensions do not
    weigh equally in that. Cloning the action measures what is actually wanted and the latent
    term shapes the representation, which is also how DextrAH-G and Miki et al. arrange it.

    Unlike the non-recurrent ``Distillation`` above, ``gradient_length`` here is a real
    truncated-BPTT window: the parent detaches the hidden state at every optimiser step, so
    gradients reach back exactly that far. It must be long enough to contain a contact event,
    since that is the only place mass and friction are visible. See the design note.
    """

    def __init__(
        self,
        policy,
        latent_loss_coef: float = 1.0,
        aux_pos_loss_coef: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.latent_loss_coef = latent_loss_coef
        self.aux_pos_loss_coef = aux_pos_loss_coef

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior = 0.0
        mean_latent = 0.0
        mean_aux = 0.0
        loss = 0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                actions, latent, aux = self.policy.act_inference(obs, return_extras=True)

                behavior_loss = self.loss_fn(actions, privileged_actions)
                latent_loss = self.loss_fn(latent, self.policy.get_latent_target(obs))
                aux_loss = self.loss_fn(aux, self.policy.get_aux_target(obs))
                loss = (
                    loss
                    + behavior_loss
                    + self.latent_loss_coef * latent_loss
                    + self.aux_pos_loss_coef * aux_loss
                )

                mean_behavior += behavior_loss.item()
                mean_latent += latent_loss.item()
                mean_aux += aux_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior /= cnt
        mean_latent /= cnt
        mean_aux /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()

        return {
            "behavior": mean_behavior,
            "latent": mean_latent,
            "object_position": mean_aux,
            "total": mean_behavior
            + self.latent_loss_coef * mean_latent
            + self.aux_pos_loss_coef * mean_aux,
        }
