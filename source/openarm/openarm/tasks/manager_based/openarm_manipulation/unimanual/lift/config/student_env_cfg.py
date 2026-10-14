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

"""Distillation environments: the lift task seen through a depth camera.

``Arm-Lift-Cube-Student`` gives the student clean depth from a fixed camera: it learns
to copy the teacher and to regress the object position before anything is taken away
from it.

Difficulty is meant to be introduced a piece at a time rather than all at once, the way
Miki et al. stage student training, with each stage resuming from the one before it.
What is still missing is camera-pose perturbation -- robustness to the calibration error
a real head-mounted camera will have -- and the DextrAH-G F.1 depth augmentations:
dropout, random pixels, stick artifacts, correlated and uncorrelated sensor noise.

The camera sits just below the arm base, looking straight ahead. That mirrors the
intended hardware -- an arm on a quadruped's back with the depth camera in its head --
and keeps the arm out of the line of sight during the approach, which is what lets the
student stay non-recurrent. See doc/design/student_distillation_design.md.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift import mdp

from .joint_pos_env_cfg import OpenArmCubeLiftEnvCfg, OpenArmCubeLiftEnvCfg_PLAY

##
# Camera and observation geometry.
##

DEPTH_WIDTH = 80
DEPTH_HEIGHT = 60
DEPTH_NEAR = 0.1
DEPTH_FAR = 1.2

CAMERA_POS = (0.0, 0.0, 0.10)
"""Level with the arm base at z = 0.10, looking down the +x axis (no tilt)."""

CAMERA_ROT = (1.0, 0.0, 0.0, 0.0)
CAMERA_CONVENTION = "world"
"""Identity in the "world" convention: forward axis +X, up axis +Z. No tilt."""

CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_FOCAL_LENGTH = 11.3
"""~85.7 deg horizontal FOV: 2 * atan(20.955 / (2 * 11.3)).

Vertical follows from the aspect ratio: 20.955 * 60 / 80 gives ~69.6 deg.
So the camera sees |y| < 0.927 d horizontally and 0.10 +/- 0.695 d vertically.
"""

STUDENT_STATE_DIM = 33
"""joint_pos 9 + joint_vel 9 + last_action 8 + target_object_position 7."""

OBJECT_POSITION_SLICE = (18, 21)
"""Where object_position sits in the teacher observation, the auxiliary head's target.

Teacher layout: joint_pos 9 | joint_vel 9 | object_position 3 | target 7 | actions 8.
"""


@configclass
class StudentObservationsCfg:
    """Two groups: what the student sees, and the privileged vector the teacher needs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Student observation: proprioception, goal, and the depth image.

        Term order fixes the layout the model slices, so the depth image must come last.
        """

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        depth = ObsTerm(
            func=mdp.depth_image_flat,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "near": DEPTH_NEAR,
                "far": DEPTH_FAR,
            },
        )

        def __post_init__(self):
            # Level 1 keeps perception ideal; higher levels are where noise comes in.
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Exactly the observation the PPO teacher was trained on -- order matters."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


def _depth_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/DepthCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=CAMERA_POS, rot=CAMERA_ROT, convention=CAMERA_CONVENTION),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=(DEPTH_NEAR, DEPTH_FAR),
        ),
        width=DEPTH_WIDTH,
        height=DEPTH_HEIGHT,
    )


@configclass
class RobotEnvCfgStudent(OpenArmCubeLiftEnvCfg):
    """The teacher's task with a clean, fixed depth camera added."""

    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _depth_camera_cfg()
        # Rendering dominates the step cost, so distillation runs far fewer envs than PPO.
        self.scene.num_envs = 256


@configclass
class RobotPlayEnvCfgStudent(OpenArmCubeLiftEnvCfg_PLAY):
    """The student task for play / visual inspection."""

    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _depth_camera_cfg()
