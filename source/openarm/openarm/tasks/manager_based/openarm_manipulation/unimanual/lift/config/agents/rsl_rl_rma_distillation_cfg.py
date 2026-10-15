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

"""Distillation config for the RMA adapter. See doc/design/arm_lift_rma_design.md."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.arm_lift_student_env_cfg import (
    PRIVILEGED_DIM,
    STUDENT_STATE_DIM,
)
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.student_env_cfg import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
)
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.agents.rsl_rl_rma_cfg import (
    LATENT_DIM,
    PRIVILEGED_CATEGORY_SLICE,
)


@configclass
class RmaStudentTeacherCfg(RslRlDistillationStudentTeacherRecurrentCfg):
    """The depth adapter, plus the frozen RMA teacher it is cloned from."""

    class_name: str = "RmaStudentTeacher"

    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"

    # The flattened observation carries a depth image; running mean/std over raw pixels is
    # not what we want, and the privileged terms are already hand-scaled in observations.py.
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False

    # Must match the PPO checkpoint: these become the teacher's policy half, which is loaded
    # from `actor.policy.*` and then frozen.
    teacher_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
    student_hidden_dims: list[int] = [256, 128, 64]  # placeholder, replaced at construction

    rnn_type: str = "gru"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1
    teacher_recurrent: bool = False

    # --- observation layout ---
    state_dim: int = STUDENT_STATE_DIM
    privileged_dim: int = PRIVILEGED_DIM
    image_height: int = DEPTH_HEIGHT
    image_width: int = DEPTH_WIDTH

    # --- must reproduce the teacher's encoder exactly, or the checkpoint will not load ---
    latent_dim: int = LATENT_DIM
    encoder_hidden_dims: list[int] = [128, 64]
    category_slice: tuple[int, int] = PRIVILEGED_CATEGORY_SLICE
    category_embed_dim: int = 32

    # --- the student's own half ---
    aux_dim: int = 3
    cnn_channels: list[int] = [16, 32, 64]
    cnn_head_dims: list[int] = [128]
    student_head_dims: list[int] = [256, 128]
    student_activation: str = "elu"


@configclass
class RmaDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "RmaDistillation"

    num_learning_epochs: int = 2
    learning_rate: float = 5.0e-4
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

    gradient_length: int = 24
    """Truncated-BPTT window, in environment steps -- 0.48 s at this control rate.

    This is the one number that has to change when a recurrent student replaces a
    feed-forward one. With the cube task's non-recurrent student the same field meant plain
    gradient accumulation and 4 was fine; with a GRU the parent detaches the hidden state at
    every optimiser step, so gradients reach back exactly this far. At 4 steps the window is
    0.08 s, which does not contain a single contact event -- and mass and friction appear
    nowhere except in how the object responds to contact, so the recurrence would never be
    given a reason to remember anything.

    Set equal to ``num_steps_per_env`` (which must be a multiple of it), the same window
    PPO's recurrent generator uses for Go2-Blind-GRU-Phase1. The hidden state itself is only
    detached, never reset, so what the GRU can carry is still the whole episode.
    """

    latent_loss_coef: float = 1.0
    aux_pos_loss_coef: float = 0.1


@configclass
class ArmLiftStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 50
    experiment_name: str = ""  # same as task name
    empirical_normalization: bool = False

    # "state" comes before "privileged" so the concatenation reproduces the teacher's actor
    # input. "aux" is a target, not an input.
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "teacher": ["state", "privileged"],
        "aux": ["aux"],
    }

    policy: RmaStudentTeacherCfg = RmaStudentTeacherCfg()
    algorithm: RmaDistillationAlgorithmCfg = RmaDistillationAlgorithmCfg()
