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

"""``Arm-Lift``: lift a primitive of unknown shape, mass and friction.

The RMA teacher (Liang et al., doc/papers/Rapid_Motor_Adaptation_for_Robotic_Manipulator_Arms.md).
The policy is split in two: an encoder that compresses the privileged description of the
object into 16 numbers, and a policy that acts on the ordinary observation plus those 16.
A depth student later learns to produce the same 16 from what a camera can see, which is
why the split exists at all -- see doc/design/arm_lift_rma_design.md.

Differences from the paper that are deliberate:

* no instance dictionary. The paper carries a learned 32-vector per object because YCB
  meshes are not describable by a bounding box. Every object here is a convex primitive and
  is fully described by its category and three half-extents, so the table would be a lookup
  of something already handed over exactly. The paper's own ablation puts the cost at 3.4
  points over 78 complex meshes; over five primitive families it should be nothing.
* no external disturbance force.
* the object's rotation is six numbers, not three. Yaw is randomised over the full circle,
  so a minimal parameterisation would cross its wrap-around every episode.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .. import mdp
from ..lift_env_cfg import ARM_INIT_NUM_LEVELS, OBJECT_POSITION_NOISE_STD
from ..object_catalog import build_catalog, catalog_spawn_cfg
from .joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

##
# Randomisation, after the RMA paper's Table 2.
##

DENSITY_RANGE = (0.5, 5.0)
"""Multiplier on the catalog's nominal 1000 kg/m^3, resampled every episode.

A 4 cm object therefore runs from about 43 g to 426 g. Wide, but it is a multiplier on
volume rather than an absolute mass, so a small object stays light relative to a large one
and the policy is learning "this material is dense", not "this object is heavy".
"""

FRICTION_RANGE = (0.50, 1.10)
"""Static friction, resampled every episode; dynamic is set equal to it. From the RMA paper.

This range was chosen to straddle the self-locking boundary of the catalog's cones, on the
reasoning that it would make success depend on a number nothing can observe. Measured, it
does not: cone success is flat across the range (80.5 / 76.6 / 77.6 / 78.0 per quarter) and
so is sphere success. See CONE_RADIUS in object_catalog.py for the figures and for what did
correlate. Friction is kept in the privileged vector because it is one dimension out of
eighteen, not because anything here has shown it to matter.
"""

LIFT_MARGIN = 0.02
"""How far above its own resting height the object counts as lifted."""

OBJECT_SPAWN_RANGE = {
    "x": (-0.10, 0.02),
    "y": (-0.18, 0.18),
    "yaw": (-math.pi, math.pi),
}
"""Carried over unchanged from the cube task, where the limits were measured.

The far edge is reach, not perception: failure to pick the cube up at all went 1.9% at
0.35-0.40 m from the base to 75.6% at 0.50-0.56 m. The lateral extent is the camera: at
x = 0.30 the 85.7 deg field of view sees |y| < 0.278.
"""


@configclass
class ArmLiftObservationsCfg:
    """What the policy sees, and what only the encoder sees.

    Two groups, concatenated in this order to form the actor's input. The privileged group
    must stay last: the model slices it off the end by width.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Available at deployment -- proprioception, the goal, and a noisy object position."""

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
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=OBJECT_POSITION_NOISE_STD, operation="add"
                ),
                bias_noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=OBJECT_POSITION_NOISE_STD, operation="abs"
                ),
            ),
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        tcp_to_object = ObsTerm(func=mdp.tcp_to_object)
        object_to_goal = ObsTerm(func=mdp.object_to_goal)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Simulator-only. The encoder compresses this; the critic reads it raw."""

        object_extent = ObsTerm(func=mdp.object_extent)
        object_mass = ObsTerm(func=mdp.object_mass)
        object_friction = ObsTerm(func=mdp.object_friction)
        object_category = ObsTerm(func=mdp.object_category)
        object_rot_6d = ObsTerm(func=mdp.object_rot_6d)
        finger_contact_force = ObsTerm(
            func=mdp.finger_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "finger_contact",
                    body_names=["openarm_left_finger", "openarm_right_finger"],
                )
            },
        )

        def __post_init__(self):
            # Terms are hand-scaled in observations.py; corrupting the quantities the
            # encoder exists to read would defeat the point.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class ArmLiftEventCfg:
    """Reset redraws the object's physics and the pose it is resting in.

    Nothing resolves the catalog here: ``object_catalog`` builds its buffers on first use,
    because the observation manager is constructed before startup events are applied and
    probes every term while resolving its dimensions.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Must follow reset_all, which puts every root back at its default pose.
    reset_object_pose = EventTerm(
        func=mdp.reset_object_stable_pose,
        mode="reset",
        params={"pose_range": OBJECT_SPAWN_RANGE},
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
class ArmLiftRewardsCfg:
    """The cube task's rewards, with the height gates made relative.

    Weights are unchanged. The only substantive edit is that "lifted" now means "``margin``
    above where this object was resting" rather than "above a fixed world z", which with a
    mixed catalog is the difference between asking every shape for the same work and asking
    tall ones for less.
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1)

    lifting_object = RewTerm(
        func=mdp.object_is_lifted_relative, params={"margin": LIFT_MARGIN}, weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_relative,
        params={"std": 0.3, "margin": LIFT_MARGIN, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_relative,
        params={"std": 0.05, "margin": LIFT_MARGIN, "command_name": "object_pose"},
        weight=5.0,
    )

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
class ArmLiftEnvCfg(OpenArmCubeLiftEnvCfg):
    """The RMA teacher task.

    Inherits the robot, its base height, the arm action and the end-effector frame from the
    cube task -- all of that was measured and none of it changes. Replaced here: the object
    (a catalog of primitives instead of one USD cube), the gripper action (continuous
    instead of binary), the observations, the events and the height gates.
    """

    observations: ArmLiftObservationsCfg = ArmLiftObservationsCfg()
    events: ArmLiftEventCfg = ArmLiftEventCfg()
    rewards: ArmLiftRewardsCfg = ArmLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Contact sensing has to be switched on at spawn time; a sensor alone reports zeros.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.finger_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_.*_finger",
            history_length=0,
            track_air_time=False,
        )

        self.actions.gripper_action = mdp.MimicJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_finger_joint.*"],
        )

        # One prim per environment, dealt round-robin from the catalog. The z here only
        # seeds default_root_state; reset_object_stable_pose computes the real height from
        # whichever stable pose the episode drew.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.05], rot=[1, 0, 0, 0]),
            spawn=catalog_spawn_cfg(build_catalog()),
        )


@configclass
class ArmLiftEnvCfg_PLAY(ArmLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Outside training the curriculum restarts at level 0 and would need hundreds of
        # resets to climb, so play would only ever show the default pose.
        self.events.reset_robot_joints.params["fixed_level"] = ARM_INIT_NUM_LEVELS
        self.curriculum.arm_init_level = None
