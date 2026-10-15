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

"""PPO config for the RMA teacher. See doc/design/arm_lift_rma_design.md."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

LATENT_DIM = 16
"""Width of ``z``, the only number in this file that is not inherited from the cube task.

45 privileged inputs compress to 16. The locomotion RMA papers use 8; this task has object
geometry to carry as well as the dynamics, so it sits at the upper end of that range. It is
also the dimension the depth student has to regress in phase 2, so it is the knob to turn
first if the adapter's loss will not come down.
"""

PRIVILEGED_CATEGORY_SLICE = (5, 10)
"""Where the category one-hot sits inside the privileged vector.

Layout, fixed by the term order in ``ArmLiftObservationsCfg.PrivilegedCfg``:
``extent 0:3 | mass 3:4 | friction 4:5 | category 5:10 | rot_6d 10:16 | contact 16:18``.
Change that order and this has to change with it.
"""


@configclass
class RmaActorCriticCfg(RslRlPpoActorCriticCfg):
    """The stock actor-critic plus the environment encoder in front of the actor."""

    class_name: str = "RmaActorCritic"

    init_noise_std: float = 1.0
    # "scalar" hands the raw parameter to Normal as its scale with nothing keeping it
    # positive; a cube run died at iteration 364 when the entropy bonus pushed it up to
    # about 2 and the policy gradient drove it back through zero. exp(log_std) cannot.
    noise_std_type: str = "log"

    # The privileged terms are scaled by hand in observations.py against ranges we set, so
    # running statistics would only add drift over the early iterations this task is
    # already known to be fragile in.
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # Unchanged from the cube teacher, which converges with them. The input grows from 36 to
    # 58 and the critic to 60; that is not a reason to resize the network around it.
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"

    # --- environment encoder ---
    privileged_group: str = "privileged"
    latent_dim: int = LATENT_DIM
    # Smaller than the policy on purpose: the encoder's input is 45 hand-picked, already
    # meaningful numbers. There is nothing to perceive, only to compress and mix, and a wide
    # MLP here would mostly memorise the five-row category table.
    encoder_hidden_dims: list[int] = [128, 64]
    category_slice: tuple[int, int] = PRIVILEGED_CATEGORY_SLICE
    category_embed_dim: int = 32


@configclass
class ArmLiftRmaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = ""  # same as task name
    empirical_normalization = False

    # The actor reads both groups -- the privileged one last, because the encoder slices it
    # off the end. The critic reads the same thing, which gives it the privileged vector raw,
    # without the bottleneck.
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy: RmaActorCriticCfg = RmaActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
