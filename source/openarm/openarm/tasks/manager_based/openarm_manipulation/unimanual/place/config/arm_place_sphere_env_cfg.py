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

"""``Arm-Place-Sphere``: ManiSkill's ``PlaceSphere-v1`` on OpenArm.

Built on ``Arm-Lift`` -- robot, base height, arm action, continuous gripper, the RMA
observation split and the arm-init curriculum are all inherited. The goal moves from a point
in the air to the floor of a bin, which is done by publishing the bin target as the
``object_pose`` command the observations already read. See
doc/design/arm_place_sphere_design.md.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...lift.config.arm_lift_env_cfg import (
    DENSITY_RANGE,
    FRICTION_RANGE,
    LIFT_MARGIN,
    OBJECT_SPAWN_RANGE,
    ArmLiftEnvCfg,
)
from ...lift.lift_env_cfg import ARM_INIT_NUM_LEVELS, CurriculumCfg
from ...lift.mdp.events import TABLE_TOP_Z
from ...lift.object_catalog import build_object_catalog, catalog_spawn_cfg
from .. import mdp
from ..bin_asset import BIN_FLOOR_HALF_THICKNESS, bin_usd_path

OBJECT_CATALOG = "sphere"
"""One 2 cm sphere, as ManiSkill. See ``build_sphere_catalog``."""

WORKSPACE_X = (0.30, 0.42)
WORKSPACE_Y = (-0.18, 0.18)
"""Where the bin may go, in the env frame (= the robot's, which sits at the env origin).

The same box the object spawns in -- ``OBJECT_SPAWN_RANGE`` around the nominal x = 0.40 --
which the lift task measured as both reachable and inside the depth camera's view.
"""

BIN_DISTANCE_RANGE = (0.08, 0.20)
"""Centre-to-centre, object to bin. The lower end keeps them apart: 2 cm of sphere plus the
bin's 3.5 cm half-diagonal is 5.5 cm."""


@configclass
class PlaceEventCfg:
    """Arm-Lift's reset events, with the bin placed once the object is."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_pose = EventTerm(
        func=mdp.reset_object_stable_pose,
        mode="reset",
        params={"pose_range": OBJECT_SPAWN_RANGE},
    )

    # Must follow reset_object_pose: it reads where the object was just put.
    reset_bin = EventTerm(
        func=mdp.reset_bin_near_object,
        mode="reset",
        params={
            "distance_range": BIN_DISTANCE_RANGE,
            "x_range": WORKSPACE_X,
            "y_range": WORKSPACE_Y,
        },
    )

    randomize_object_physics = EventTerm(
        func=mdp.randomize_object_physics,
        mode="reset",
        params={"density_range": DENSITY_RANGE, "friction_range": FRICTION_RANGE},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_arm_joints_by_level,
        mode="reset",
        params={
            "max_range": 0.2,
            "num_levels": ARM_INIT_NUM_LEVELS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint.*"]),
        },
    )


@configclass
class PlaceRewardsCfg:
    """Arm-Lift's six terms with the goal moved into the bin, and four for putting it down.

    Isaac Lab sums terms where ManiSkill overwrites by stage, so the weights are set so that
    each stage's best per-step total beats the previous one's:

        reaching only                         ~ 1
        lifted, held 3 cm above the bin       ~ 33   (1.1 + 15 + 14.4 + 2.3)
        in the bin, still held                ~ 42   (lift is lost; in_bin 20 covers it)
        released, at rest                     ~ 67   (+ released 10 + static 5 + success 10)
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1)

    lifting_object = RewTerm(
        func=mdp.object_is_lifted_relative, params={"margin": LIFT_MARGIN}, weight=15.0
    )

    place_tracking = RewTerm(
        func=mdp.place_tracking, params={"std": 0.3, "margin": LIFT_MARGIN}, weight=16.0
    )

    place_tracking_fine = RewTerm(
        func=mdp.place_tracking, params={"std": 0.05, "margin": LIFT_MARGIN}, weight=5.0
    )

    object_in_bin = RewTerm(func=mdp.object_in_bin_reward, weight=20.0)

    released_in_bin = RewTerm(
        func=mdp.released_in_bin,
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"])},
        weight=10.0,
    )

    static_in_bin = RewTerm(func=mdp.static_in_bin, weight=5.0)

    place_success = RewTerm(func=mdp.place_success_reward, weight=10.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
            )
        },
    )


@configclass
class PlaceCurriculumCfg(CurriculumCfg):
    """The smoothness ramps unchanged; the arm-init level promoted on placement instead."""

    arm_init_level = CurrTerm(func=mdp.arm_init_level_place, params={"margin": LIFT_MARGIN})

    place_success = CurrTerm(func=mdp.episode_place_success)


@configclass
class ArmPlaceSphereEnvCfg(ArmLiftEnvCfg):
    """The Arm-Lift teacher with a bin to put a sphere in."""

    object_catalog: str = OBJECT_CATALOG
    """Read by ``object_catalog()`` in lift/mdp/events.py."""

    events: PlaceEventCfg = PlaceEventCfg()
    rewards: PlaceRewardsCfg = PlaceRewardsCfg()
    curriculum: PlaceCurriculumCfg = PlaceCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # One more second than Arm-Lift for lowering and letting go.
        self.episode_length_s = 6.0

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.05], rot=[1, 0, 0, 0]),
            spawn=catalog_spawn_cfg(build_object_catalog(OBJECT_CATALOG)),
        )

        # Kinematic: it is placed at reset and nothing the arm does can move it.
        self.scene.bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bin",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.4, 0.2, TABLE_TOP_Z + BIN_FLOOR_HALF_THICKNESS], rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=bin_usd_path(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
            ),
        )

        # Finger-on-object force only. The existing finger_contact is the net force, which
        # includes the table and the bin, and so cannot say whether the object is held.
        # One sensor per finger -- see FINGER_OBJECT_SENSORS in place/mdp/rewards.py.
        for side in ("left", "right"):
            setattr(
                self.scene,
                f"{side}_finger_object_contact",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_finger",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    history_length=0,
                    track_air_time=False,
                ),
            )

        self.commands.object_pose = mdp.BinTargetCommandCfg()


@configclass
class ArmPlaceSphereEnvCfg_PLAY(ArmPlaceSphereEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # As Arm-Lift's PLAY: show the level the policy was trained up to.
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None
