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

"""``Arm-Lift-Student``: the same task, seen through a depth camera.

Four observation groups rather than two, because the student and the teacher need
overlapping but different slices and rsl-rl assembles each from a list of groups:

    policy      x_t + the depth image        what the student sees
    state       x_t                          shared prefix, fed to the teacher's policy half
    privileged  e, s_t                       the teacher's encoder input
    aux         the clean object position    a regression target, nothing reads it as input

``state`` is listed first in the teacher's groups so the concatenation reproduces the
teacher's own actor input exactly; the student's action is produced by applying the same
policy half to ``state`` and its own predicted latent, so the two have to line up.

Perception is kept ideal here -- no observation noise, no depth augmentation, a camera that
is exactly where it is said to be. Difficulty is meant to come in a stage at a time, the way
Miki et al. arrange student training, each stage resuming from the one before. Still missing:
camera-pose perturbation for the calibration error a real head-mounted camera will have, and
the DextrAH-G F.1 depth augmentations (dropout, random pixels, stick artifacts, correlated
and uncorrelated sensor noise).
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .. import mdp
from .arm_lift_env_cfg import ArmLiftEnvCfg, ArmLiftObservationsCfg
from .student_env_cfg import DEPTH_FAR, DEPTH_NEAR, _depth_camera_cfg

STUDENT_STATE_DIM = 42
"""joint_pos 9 + joint_vel 9 + object_position 3 + target 7 + tcp_to_object 3
+ object_to_goal 3 + actions 8. Must match ArmLiftObservationsCfg.PolicyCfg."""

PRIVILEGED_DIM = 18
"""extent 3 + mass 1 + friction 1 + category 5 + rot_6d 6 + contact 2."""


@configclass
class ArmLiftStudentObservationsCfg:
    @configclass
    class StateCfg(ArmLiftObservationsCfg.PolicyCfg):
        """The teacher's observable half, uncorrupted.

        Distillation reads this as the teacher's input, so corruption is off: the labels
        the student imitates should be the teacher's best behaviour, not its behaviour under
        a noise draw the student cannot see. The student is fed the same tensor, which keeps
        its policy-half input identical to the teacher's.
        """

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    @configclass
    class PolicyCfg(StateCfg):
        """What the student actually sees. The depth image must stay last -- the model
        slices it off the end by width."""

        depth = ObsTerm(
            func=mdp.depth_image_flat,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "near": DEPTH_NEAR,
                "far": DEPTH_FAR,
            },
        )

    @configclass
    class PrivilegedCfg(ArmLiftObservationsCfg.PrivilegedCfg):
        """Unchanged from the teacher task; this is what the encoder was trained on."""

    @configclass
    class AuxCfg(ObsGroup):
        """Regression target only. Never an input to anything."""

        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    state: StateCfg = StateCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    aux: AuxCfg = AuxCfg()


@configclass
class ArmLiftStudentEnvCfg(ArmLiftEnvCfg):
    observations: ArmLiftStudentObservationsCfg = ArmLiftStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _depth_camera_cfg()
        # 128, where the cube task's student ran 256. The binding constraint is no longer
        # rendering throughput but the backward graph: with a 24-step BPTT window the CNN's
        # activations are held for every step, about 0.67 MB per image, so 256 environments
        # need roughly 4 GB of graph on top of the renderer's ~6 GB and overflow a 16 GB
        # card. Halving the environments halves that. The history length is what the
        # recurrence needs; the batch width is not.
        self.scene.num_envs = 128


@configclass
class ArmLiftStudentEnvCfg_PLAY(ArmLiftStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        from ..lift_env_cfg import ARM_INIT_NUM_LEVELS

        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None
