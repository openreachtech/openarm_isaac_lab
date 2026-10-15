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

"""JIT / ONNX export for the RMA depth student, hidden state and all.

What gets exported is the whole deployable path -- the depth CNN, the GRU, the head that
produces the latent, and the teacher's policy half that turns it into actions. The teacher's
*encoder* is not part of it: that is the thing this student exists to replace.

Two departures from Isaac Lab's exporters, both forced:

* they call ``torch.jit.script``, which cannot handle rsl-rl's ``Memory``. It declares
  ``hidden_state`` as a union so one class can wrap both GRU and LSTM, and TorchScript then
  refuses to pass it to ``GRU.forward``, whose ``hx`` is ``Optional[Tensor]``. Tracing has
  no such problem, and the graph here has no data-dependent control flow for tracing to get
  wrong.
* the hidden state is an explicit input *and* output rather than module state. A deployed
  policy is called frame by frame from outside, so the caller has to own the recurrence:
  pass zeros on the first frame of an episode and feed each output back in. Hiding it inside
  the graph would make the exported model unusable.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from openarm.assets.models.rma_student import RmaDepthStudent


class _ExportWrapper(nn.Module):
    """``(obs, hidden)`` in, ``(action, object_position, hidden)`` out, on CPU."""

    def __init__(
        self,
        student: RmaDepthStudent,
        policy_half: nn.Module,
        normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.student = copy.deepcopy(student).cpu().eval()
        self.policy = copy.deepcopy(policy_half).cpu().eval()
        self.normalizer = (
            copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else nn.Identity()
        )
        # Used directly rather than through Memory, so the hidden state stays in the signature.
        self.rnn = self.student.memory.rnn

    @property
    def num_obs(self) -> int:
        return self.student.num_obs

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return self.rnn.num_layers, self.rnn.hidden_size

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self.normalizer(obs)
        state, image = self.student.split_obs(obs)
        memory_out, hidden = self.rnn(state.unsqueeze(0), hidden)
        features = self.student.head(
            torch.cat([self.student.cnn(image), memory_out.squeeze(0)], dim=-1)
        )
        latent = self.student.latent_head(features)
        action = self.policy(torch.cat([state, latent], dim=-1))
        return action, self.student.aux_head(features), hidden


def export_rma_student(
    student: RmaDepthStudent,
    policy_half: nn.Module,
    path: str,
    normalizer: nn.Module | None = None,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
    opset_version: int = 14,
) -> None:
    """Write a traced TorchScript module and an ONNX graph for the depth student.

    ``opset_version`` is 14 rather than the 11 the feed-forward student uses, because that is
    where ONNX's GRU gained the behaviour PyTorch's exporter needs.
    """
    os.makedirs(path, exist_ok=True)
    wrapper = _ExportWrapper(student, policy_half, normalizer)
    num_layers, hidden_dim = wrapper.hidden_shape
    dummy_obs = torch.zeros(1, wrapper.num_obs)
    dummy_hidden = torch.zeros(num_layers, 1, hidden_dim)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_obs, dummy_hidden))
    traced.save(os.path.join(path, jit_filename))

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_hidden),
        os.path.join(path, onnx_filename),
        export_params=True,
        opset_version=opset_version,
        input_names=["obs", "hidden_in"],
        output_names=["actions", "object_position", "hidden_out"],
        dynamic_axes={
            "obs": {0: "batch"},
            "hidden_in": {1: "batch"},
            "actions": {0: "batch"},
            "object_position": {0: "batch"},
            "hidden_out": {1: "batch"},
        },
    )
    print(
        f"[INFO] Exported RMA student to {path} ({jit_filename}, {onnx_filename});"
        f" hidden state is ({num_layers}, batch, {hidden_dim}), caller-owned."
    )
