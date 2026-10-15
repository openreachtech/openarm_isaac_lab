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

"""Project-specific wrappers around rsl-rl's runners.

rsl-rl picks its policy and algorithm classes with ``eval()`` evaluated inside its own
runner module, so a custom class cannot simply be named in the config -- it has to be
bound into that module's namespace first. These subclasses do that binding and nothing
else.
"""

from __future__ import annotations

from rsl_rl.runners import DistillationRunner, OnPolicyRunner


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

    if train_cfg.get("policy", {}).get("class_name") == "RmaStudentTeacher":
        from openarm.assets.models.modules.rma_student_teacher import RmaStudentTeacher

        distillation_runner_module.RmaStudentTeacher = RmaStudentTeacher

    if train_cfg.get("algorithm", {}).get("class_name") == "RmaDistillation":
        from openarm.assets.models.modules.distillation import RmaDistillation

        distillation_runner_module.RmaDistillation = RmaDistillation


class OpenArmDistillationRunner(DistillationRunner):
    """DistillationRunner that knows about the depth student and auxiliary loss."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_distillation_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)


def _register_on_policy_classes(train_cfg: dict) -> None:
    """Expose the RMA actor-critic to ``OnPolicyRunner``.

    Same mechanism as the distillation side: ``OnPolicyRunner`` resolves
    ``policy.class_name`` with ``eval()`` in the scope of ``rsl_rl.runners.on_policy_runner``
    (``on_policy_runner.py:416``), so the class is injected there.
    """
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    if train_cfg.get("policy", {}).get("class_name") == "RmaActorCritic":
        from openarm.assets.models.rma_actor import RmaActorCritic

        on_policy_runner_module.RmaActorCritic = RmaActorCritic


class OpenArmOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that also knows about the RMA actor-critic.

    A drop-in for the stock runner: with any other ``policy.class_name`` the registration
    hook does nothing, so the existing cube task behaves identically.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        _register_on_policy_classes(train_cfg)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
