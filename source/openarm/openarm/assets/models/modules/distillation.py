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
