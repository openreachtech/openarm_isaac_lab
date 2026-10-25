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
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
from ..bin_asset import DEEP_BIN, HAND_BOX, SHALLOW_BIN, BinCfg, bin_usd_path

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

    bin_geometry: BinCfg = SHALLOW_BIN
    """Read by the bin's spawn, its reset event and every bin predicate."""

    place_tolerance: float | None = None
    """How close to the bin's centre counts as placed (``object_on_target``). None: anywhere
    in the pocket."""

    require_hand_placement: bool = False
    """Pay the bin rewards only for an object still held or let go of low enough to count as
    placing, not dropped in (``placement_valid`` in place/mdp/release.py)."""

    success_requires_rest: bool = True
    """Whether ``place_success`` also asks for the object to be at rest."""

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
            spawn=catalog_spawn_cfg(build_object_catalog(self.object_catalog)),
        )

        # Kinematic: it is placed at reset and nothing the arm does can move it.
        self.scene.bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bin",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.4, 0.2, TABLE_TOP_Z + self.bin_geometry.floor_half_thickness], rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=bin_usd_path(self.bin_geometry),
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


##
# Deep bin
##


@configclass
class ArmPlaceSphereDeepEnvCfg(ArmPlaceSphereEnvCfg):
    """Walls twice the sphere's diameter, pocket 1.5x as wide -- see ``DEEP_BIN``.

    The only reward change is the coarse tracking term, which aims over the rim instead of
    through the wall. Everything else, the weights included, is the shallow task's.
    """

    bin_geometry: BinCfg = DEEP_BIN

    def __post_init__(self):
        super().__post_init__()
        self.rewards.place_tracking = RewTerm(
            func=mdp.place_tracking_over_rim, params={"std": 0.3, "margin": LIFT_MARGIN}, weight=16.0
        )


@configclass
class ArmPlaceSphereDeepEnvCfg_PLAY(ArmPlaceSphereDeepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None


##
# Hand-sized box
##

BOX_BIN_DISTANCE_RANGE = (0.22, 0.30)
"""Pushed out with the box. Its half-diagonal is 14.8 cm, and the fingers closing on the
object reach 5 cm from its centre, so below about 22 cm they would meet the box on the way
in. Deep failed most where the bin was close (96.5% at 8-11 cm against 99% beyond)."""

BOX_WORKSPACE_Y = (-0.28, 0.28)
"""The box needs more room than the lift task's +-0.18 to sit 22-30 cm from the object.
The far corner, (0.42, 0.28), is 0.50 m from the base; the box centre stays inside 0.51."""

SPHERE_ROLLING_DAMPING = 5.0
"""Angular damping (1/s) on the sphere, standing in for the rolling resistance PhysX lacks.

Contact is a point, as with the spin about the vertical, so nothing slows a rolling sphere
and any speed left at release carries it until a wall. With a 1 cm placement tolerance and
a 1 cm/s "static" threshold that matters: unchecked, 5 mm/s for six seconds is 3 cm. On a
rolling sphere this damping slows the centre at about 2/7 of the rate, so 1 cm/s comes to
rest within roughly 7 mm. Too small to matter while the sphere is held."""


@configclass
class BoxRewardsCfg:
    """Staged, each stage's best per-step total at or above the one before:

        reach                                                    ~ 1
        lift        carrying                                     ~ 16
        carry       carrying, over the box centre (xy only)      ~ 32
        lower       held over the box, at the floor centre       ~ 52
        release     put down by hand at the centre               ~ 112

    "Carrying" and "held" both mean gripped *or put down by hand* (``held_or_placed``), so
    letting go keeps everything holding paid and the release term is a pure bonus -- graded
    over 5 cm, so an imprecise release still gains. An object dropped in from above, or left
    on top of a wall, is neither, and earns none of the box terms.
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1)

    lifting_object = RewTerm(func=mdp.carrying_reward, params={"margin": LIFT_MARGIN}, weight=15.0)

    carry_to_box = RewTerm(func=mdp.carry_to_box, params={"std": 0.3, "margin": LIFT_MARGIN}, weight=16.0)

    # Two kernels: the wide one reaches a sphere held high over the box -- the staged
    # rewards are otherwise flat from there down to the rim, and the policy sat at ~49 cm --
    # and the narrow one does the centring at the floor.
    lower_to_target = RewTerm(func=mdp.lower_to_target, params={"std": 0.3}, weight=10.0)

    lower_to_target_fine = RewTerm(func=mdp.lower_to_target, params={"std": 0.05}, weight=10.0)

    released_on_target = RewTerm(func=mdp.released_on_target, params={"radius": 0.05}, weight=60.0)

    # Clamped, as joint_vel below: see place/mdp/terminations.py. 8 action dims each
    # changing by ~3.5 between steps is already far outside anything smooth.
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, params={"max_value": 100.0}, weight=-1e-4)

    # Clamped: see place/mdp/terminations.py. 100 rad^2/s^2 is ~4x what nine joints at their
    # velocity limits would give, so it only ever bites on a blow-up.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clamped,
        weight=-1e-4,
        params={
            "max_value": 100.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
            ),
        },
    )


@configclass
class ArmPlaceSphereBoxEnvCfg(ArmPlaceSphereEnvCfg):
    """A box the hand fits into, the sphere asked for within 1 cm of its centre, and placed by
    hand.

    A precise target alone did not stop the policy dropping the sphere from above the rim
    (2026-10-22, model_1000: 90% off target, 58% of releases from over 10 cm), so the box
    rewards also require the sphere to have been put down by hand (``require_hand_placement``).
    Success does not ask for the sphere to be at rest: on target and let go of is enough.

    Kept as the sphere counterpart of ArmPlaceCubeBoxEnvCfg; it has not been trained since
    the release fixes went in, and before them it never let go of the sphere.
    """

    bin_geometry: BinCfg = HAND_BOX
    place_tolerance: float | None = 0.01
    require_hand_placement: bool = True
    success_requires_rest: bool = False

    rolling_damping: float | None = SPHERE_ROLLING_DAMPING
    """Angular damping on the object; None leaves PhysX's default."""

    rewards: BoxRewardsCfg = BoxRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_release_tracker = EventTerm(func=mdp.reset_release_tracker, mode="reset")
        # The arm's velocity limits are 2.2-2.6 rad/s; anything near 50 is the solver, not
        # the policy.
        self.terminations.physics_blowup = DoneTerm(
            func=mdp.physics_blowup, params={"max_joint_vel": 50.0, "max_object_speed": 5.0}
        )
        self.events.reset_bin.params["distance_range"] = BOX_BIN_DISTANCE_RANGE
        self.events.reset_bin.params["y_range"] = BOX_WORKSPACE_Y
        if self.rolling_damping is not None:
            self.scene.object.spawn.assets_cfg[0].rigid_props.angular_damping = self.rolling_damping


@configclass
class ArmPlaceSphereBoxEnvCfg_PLAY(ArmPlaceSphereBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None


##
# Cube in the hand-sized box
##


RELEASE_BONUS_RADIUS = 0.02
"""Where the release bonus falls to zero, for the cube."""


@configclass
class ArmPlaceCubeBoxEnvCfg(ArmPlaceSphereBoxEnvCfg):
    """The box task with a 4 cm cube instead of the sphere.

    With the sphere the policy never let go: 87% of episodes ended gripping it on the floor,
    centred to 0.6 cm, and the few releases saw it end 5-11 cm out at a wall. The cube was
    meant to take rolling out of that, and does stay where it is put -- but on its own it
    changed nothing (76% still gripping; 19% parked on a wall top). What made the policy let
    go was the reward: holding rewards that survive a release by hand (``held_or_placed``)
    and a graded release bonus, together 62.5% within 1 cm; with the bonus sharpened to 2 cm,
    84.5%, and the policy now goes back and corrects a misplacement (2026-10-23).
    """

    object_catalog: str = "cube"
    rolling_damping: float | None = None

    def __post_init__(self):
        super().__post_init__()
        # Sharper than BoxRewardsCfg's 5 cm. At 5 cm, moving a 1.5 cm miss to 0.5 cm was
        # worth only 12 per step and the policy never went back to correct one (2026-10-23:
        # 62.5% within 1 cm, misses a median 1.5 cm out). At 2 cm a 1.5 cm miss pays a quarter
        # of a centred one: 84.5%, misses a median 1.2 cm out, and corrections seen in play.
        self.rewards.released_on_target.params["radius"] = RELEASE_BONUS_RADIUS


@configclass
class ArmPlaceCubeBoxEnvCfg_PLAY(ArmPlaceCubeBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None
