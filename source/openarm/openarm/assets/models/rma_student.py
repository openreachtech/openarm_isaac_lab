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

"""The RMA adapter: depth and a history of proprioception, in; the teacher's latent, out.

Liang et al. eq. (3). The teacher's encoder is replaced by something that can run on
hardware, and the policy half is reused unchanged.

Two structural choices, both argued in doc/design/arm_lift_rma_design.md:

* the depth features bypass the recurrence. The image carries instantaneous geometry --
  what shape is in front of the gripper -- and a hidden state adds nothing to it, while
  carrying 128 visual features through a GRU makes the recurrence much harder to train.
* the recurrence is necessary. The cube task's student was single-frame, justified by the
  fact that ``object_position`` is either visible or implied by the joint angles once the
  object is grasped. The quantities here are mass and friction, which appear only in how
  the object responded to being pushed, and a single frame cannot show a response.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import Memory

from openarm.assets.models.depth_student import DepthCNN, _mlp


class RmaDepthStudent(nn.Module):
    """``[x_t | flattened depth]`` in, ``(latent, object position)`` out."""

    def __init__(
        self,
        state_dim: int,
        image_height: int,
        image_width: int,
        latent_dim: int,
        aux_dim: int = 3,
        cnn_channels: list[int] = [16, 32, 64],
        cnn_head_dims: list[int] = [128],
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        head_dims: list[int] = [256, 128],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.image_dim = self.image_height * self.image_width

        self.cnn = DepthCNN(self.image_height, self.image_width, cnn_channels, cnn_head_dims)
        # GRU rather than LSTM, one layer: the same choice unitree_rl_lab's
        # Go2-Blind-GRU-Phase1 made, and Miki et al. before it, at three quarters of the
        # parameters. 128 rather than Go2's 256 because the depth image hands this network
        # the geometry outright -- the recurrence is only here for mass and friction.
        self.memory = Memory(self.state_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

        self.head, head_out = _mlp(self.cnn.out_dim + rnn_hidden_dim, head_dims, activation)
        self.latent_head = nn.Linear(head_out, latent_dim)
        # Kept from the cube task's student, where it measurably helped. Its target is the
        # *clean* object position, which is not part of the privileged vector the teacher
        # reads, so the student task supplies it as a group of its own.
        self.aux_head = nn.Linear(head_out, aux_dim)

    @property
    def num_obs(self) -> int:
        return self.state_dim + self.image_dim

    def split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = obs[..., : self.state_dim]
        image = obs[..., self.state_dim : self.state_dim + self.image_dim]
        return state, image.reshape(-1, 1, self.image_height, self.image_width)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, image = self.split_obs(obs)
        # Memory returns (1, N, H) in single-step mode; the leading time axis is dropped.
        memory_out = self.memory(state).squeeze(0)
        features = self.head(torch.cat([self.cnn(image), memory_out], dim=-1))
        return self.latent_head(features), self.aux_head(features)
