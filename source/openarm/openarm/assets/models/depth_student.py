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

"""Depth student policy for teacher-student distillation.

Follows DextrAH-G (appendix F.3) with two deliberate departures, both argued in
``doc/design/student_distillation_design.md``:

* no recurrent layer -- the privileged quantity the student has to recover
  (``object_position``) is either visible or, once the cube is grasped, implied by the
  joint angles, so there is nothing for a hidden state to carry;
* no separate encoders for the proprioceptive and goal blocks -- 33 raw dimensions do
  not need a 512-wide encoder when the teacher itself maps 36 raw dimensions straight
  to actions through ``[256, 128, 64]``.

The auxiliary head predicts the object position, exactly the input the teacher has and
the student does not. It is supervised against the simulator's ground truth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "lrelu": nn.LeakyReLU, "tanh": nn.Tanh}


def _mlp(in_dim: int, hidden_dims: list[int], activation: str) -> tuple[nn.Sequential, int]:
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    dim = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(dim, h), act()]
        dim = h
    return nn.Sequential(*layers), dim


class DepthCNN(nn.Module):
    """Three conv/max-pool stages followed by a 2-layer MLP (DextrAH-G F.3)."""

    def __init__(
        self,
        height: int,
        width: int,
        channels: list[int] = [16, 32, 64],
        head_dims: list[int] = [128, 128],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # Derive the flattened width from a dry run rather than hand-computing the pooling.
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 1, height, width)).flatten(1).shape[-1]

        self.head, self.out_dim = _mlp(flat_dim, head_dims, "relu")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (N, 1, H, W) -> (N, out_dim)."""
        return self.head(self.conv(image).flatten(1))


class DepthStudentPolicy(nn.Module):
    """[proprio | goal | flattened depth] -> (action, object position estimate)."""

    def __init__(
        self,
        state_dim: int,
        image_height: int,
        image_width: int,
        num_actions: int,
        aux_dim: int = 3,
        cnn_channels: list[int] = [16, 32, 64],
        cnn_head_dims: list[int] = [128, 128],
        trunk_dims: list[int] = [256, 128, 64],
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.image_dim = self.image_height * self.image_width

        self.cnn = DepthCNN(self.image_height, self.image_width, cnn_channels, cnn_head_dims)
        self.trunk, trunk_out = _mlp(self.state_dim + self.cnn.out_dim, trunk_dims, activation)
        self.action_head = nn.Linear(trunk_out, num_actions)
        self.aux_head = nn.Linear(trunk_out, aux_dim)

    @property
    def num_obs(self) -> int:
        return self.state_dim + self.image_dim

    def split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice the flat observation into the state block and the depth image."""
        state = obs[..., : self.state_dim]
        image = obs[..., self.state_dim : self.state_dim + self.image_dim]
        image = image.reshape(-1, 1, self.image_height, self.image_width)
        return state, image

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, image = self.split_obs(obs)
        latent = self.trunk(torch.cat([state, self.cnn(image)], dim=-1))
        return self.action_head(latent), self.aux_head(latent)
