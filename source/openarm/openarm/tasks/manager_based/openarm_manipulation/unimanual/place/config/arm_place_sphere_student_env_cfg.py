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

"""``Arm-Place-Sphere-Student``: the place task through the depth camera, distilled from the
``Arm-Place-Sphere`` teacher.

Everything is ``Arm-Lift-Student``'s. The place teacher keeps Arm-Lift's observation classes
-- same terms, same widths, the bin target arriving through the ``object_pose`` command --
so the four student groups (``policy``, ``state``, ``privileged``, ``aux``), the depth camera
and the distillation runner all carry over without a change.

Perception stays as ideal as in Arm-Lift-Student. In particular the bin target reaches the
student exactly, inside ``state``, as the lift goal does there; on hardware the bin would
have to be found in the depth image too.
"""

from isaaclab.utils import configclass

from ...lift.config.arm_lift_student_env_cfg import ArmLiftStudentObservationsCfg
from ...lift.config.student_env_cfg import _depth_camera_cfg
from ...lift.lift_env_cfg import ARM_INIT_NUM_LEVELS
from .arm_place_sphere_env_cfg import ArmPlaceSphereEnvCfg


@configclass
class ArmPlaceSphereStudentEnvCfg(ArmPlaceSphereEnvCfg):
    observations: ArmLiftStudentObservationsCfg = ArmLiftStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = _depth_camera_cfg()
        # 128 for the same reason as Arm-Lift-Student: the 24-step BPTT graph through the CNN.
        self.scene.num_envs = 128


@configclass
class ArmPlaceSphereStudentEnvCfg_PLAY(ArmPlaceSphereStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None
