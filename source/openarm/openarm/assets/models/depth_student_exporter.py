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

"""JIT / ONNX export for the depth student.

Isaac Lab's ``export_policy_as_jit`` / ``export_policy_as_onnx`` index the actor as
``self.actor[0].in_features``, which assumes a plain ``nn.Sequential`` MLP.
:class:`DepthStudentPolicy` is not subscriptable, so it needs its own exporter.

Both outputs are exported. The action is what a controller consumes; the object-position
estimate is the auxiliary head, which DextrAH-G feeds to a state machine during
deployment (appendix G).
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from openarm.assets.models.depth_student import DepthStudentPolicy


class _ExportWrapper(nn.Module):
    """Normalizer + student on CPU, returning ``(action, object_position)``."""

    def __init__(self, student: DepthStudentPolicy, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.student = copy.deepcopy(student).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else nn.Identity()

    @property
    def num_obs(self) -> int:
        return self.student.num_obs

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(self.normalizer(obs))


def export_depth_student(
    student: DepthStudentPolicy,
    path: str,
    normalizer: nn.Module | None = None,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
    opset_version: int = 11,
) -> None:
    """Write a traced TorchScript module and an ONNX graph for the depth student."""
    os.makedirs(path, exist_ok=True)
    wrapper = _ExportWrapper(student, normalizer)
    dummy = torch.zeros(1, wrapper.num_obs)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)
    traced.save(os.path.join(path, jit_filename))

    torch.onnx.export(
        wrapper,
        dummy,
        os.path.join(path, onnx_filename),
        export_params=True,
        opset_version=opset_version,
        input_names=["obs"],
        output_names=["actions", "object_position"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}, "object_position": {0: "batch"}},
    )
    print(f"[INFO] Exported depth student to {path} ({jit_filename}, {onnx_filename})")
