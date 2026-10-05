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

"""Project-specific wrapper around rsl-rl's DistillationRunner."""

from __future__ import annotations

from rsl_rl.runners import DistillationRunner


def _register_distillation_classes(train_cfg: dict) -> None:
    """Expose the depth student / auxiliary-loss classes to the runner.

    rsl-rl resolves ``policy.class_name`` and ``algorithm.class_name`` with ``eval()`` in
    the scope of ``rsl_rl.runners.distillation_runner``, so the classes have to be
    injected there rather than passed in. Named after the module they shadow so the cfg
    reads like stock rsl-rl.
    """
    import rsl_rl.runners.distillation_runner as distillation_runner_module

    if train_cfg.get("policy", {}).get("class_name") == "DepthStudentTeacher":
        from openarm.assets.models.modules.student_teacher import DepthStudentTeacher

        distillation_runner_module.DepthStudentTeacher = DepthStudentTeacher

    if train_cfg.get("algorithm", {}).get("class_name") == "Distillation":
        from openarm.assets.models.modules.distillation import Distillation

        distillation_runner_module.Distillation = Distillation


class OpenArmDistillationRunner(DistillationRunner):
    """DistillationRunner that knows about the depth student and auxiliary loss."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_distillation_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
