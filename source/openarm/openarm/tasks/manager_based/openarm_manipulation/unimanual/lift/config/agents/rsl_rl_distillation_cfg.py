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

"""Depth-student distillation configs. See doc/design/student_distillation_design.md."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.student_env_cfg import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    OBJECT_POSITION_SLICE,
    STUDENT_STATE_DIM,
)


@configclass
class DepthStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Depth student + the frozen PPO teacher it is cloned from."""

    class_name: str = "DepthStudentTeacher"

    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"

    # The flattened observation carries a depth image; running mean/std over raw pixels
    # is not what we want, so the model scales depth to [0, 1] itself.
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False

    # Must match the PPO teacher checkpoint: the parent class rebuilds the teacher as a
    # plain MLP and loads `actor.*` into it.
    teacher_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"

    # Consumed by the parent's placeholder student, which is then discarded.
    student_hidden_dims: list[int] = [256, 128, 64]

    # --- depth student ---
    state_dim: int = STUDENT_STATE_DIM
    image_height: int = DEPTH_HEIGHT
    image_width: int = DEPTH_WIDTH
    aux_obs_slice: tuple[int, int] = OBJECT_POSITION_SLICE
    cnn_channels: list[int] = [16, 32, 64]
    cnn_head_dims: list[int] = [128, 128]
    trunk_dims: list[int] = [256, 128, 64]
    student_activation: str = "elu"


@configclass
class DepthDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Behavior cloning + object-position regression."""

    class_name: str = "Distillation"

    num_learning_epochs: int = 2
    learning_rate: float = 5.0e-4
    # The student is not recurrent, so this is the effective minibatch
    # (gradient_length * num_envs), not a BPTT window. num_steps_per_env must be a
    # multiple of it.
    gradient_length: int = 4
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

    aux_pos_loss_coef: float = 0.1
    """DextrAH-G's beta: weight on the object-position term."""


@configclass
class OpenArmLiftCubeDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 50
    experiment_name: str = ""  # same as task name
    empirical_normalization: bool = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "teacher": ["teacher"]}
    policy: DepthStudentTeacherCfg = DepthStudentTeacherCfg()
    algorithm: DepthDistillationAlgorithmCfg = DepthDistillationAlgorithmCfg()
