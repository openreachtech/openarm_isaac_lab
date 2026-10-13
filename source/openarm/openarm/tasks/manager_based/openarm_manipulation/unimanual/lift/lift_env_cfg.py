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


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import math

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

OBJECT_SCALE_RANGE = (0.72, 0.88)
"""Absolute ``xformOp:scale`` range for the cube; nominal is 0.80.

Note this *replaces* the 0.8 in the spawn config rather than multiplying it, so the
range is absolute. Measured edge lengths are 3.72-4.68 cm against a nominal 4.20 cm, i.e.
about +/-11%.

Deliberately mild. The limits it has to stay inside:
  grasp   face diagonal must clear the 0.088 m gripper aperture; at 0.88 that is
          0.065 m, still comfortable.
  lift    ``minimal_height = 0.04`` is a fixed threshold and the cube rests at
          edge/2, i.e. 0.019-0.023 m, so the gate stays meaningful at both ends.
  vision  a smaller cube means fewer pixels for the student; at 0.72 and 0.5 m away
          it spans about 5 px.

Isaac Lab refuses this term outright unless ``replicate_physics`` is False, because a
replicated scene shares one collision prototype across environments -- the visuals would
vary while the physics did not.

Applied in the "prestartup" mode, so each environment gets one fixed size for the whole
run rather than a fresh one per episode. Across 2048 environments that is still a wide
spread of sizes within a batch.
"""

ARM_INIT_MAX_RANGE = 0.2
"""The +- joint offset at the top level, in radians.

Measured at reset, +-0.2 rad spreads the TCP over 8.4 x 18.6 x 23.2 cm (5-95% widths)
and up to 22 cm from the default pose -- several times the cube. Nothing collides: the
lowest link stays at the base height out to +-0.4, and only past +-0.5 does the arm
start dipping toward the table.

Set generously on purpose. The curriculum cannot climb past what the policy can hold,
so a high ceiling costs nothing and the mean level reports where the real limit is. The
earlier ceiling of 0.1 was reached by both seeds with room to spare, which told us only
that it was too low.
"""

ARM_INIT_NUM_LEVELS = 10
"""Levels between "always the default pose" and ARM_INIT_MAX_RANGE, so 0.02 rad each."""


OBJECT_POSITION_NOISE_STD = 0.02
"""Std (m) of the noise on the teacher's object-position observation.

DextrAH-G trains its privileged teacher on a corrupted object pose so the policy stays
usable once a depth student -- whose position estimate is only accurate to a few cm --
drives it (paper section 3.2 "Pose Noise", appendix E.4, where
sigma_xyz,uncorr = sigma_xyz,corr = 0.02 m). The same value is used here.

Two components, as in the paper: one resampled every step, one sampled per episode and
held. The per-episode bias is what makes a *systematic* offset survivable; step noise
alone would simply average out.

Only the teacher sees this. Distillation reads the teacher group with
``enable_corruption = False``, so the labels the student imitates stay clean.
"""


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.4),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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
                # per-step, zero-mean
                noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=OBJECT_POSITION_NOISE_STD, operation="add"
                ),
                # per-episode bias: "abs" replaces the stored bias with a fresh sample on
                # reset, rather than adding to it (which would random-walk across episodes)
                bias_noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=OBJECT_POSITION_NOISE_STD, operation="abs"
                ),
            ),
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # x in [0.30, 0.42], y in [-0.18, 0.18] around a nominal (0.40, 0.00).
            #
            # The far edge is trimmed because the arm cannot reliably grasp there. The
            # TCP reaches 0.616 m from the base, which at 0.13 m above the table leaves
            # 0.602 m horizontally, and a top-down grasp needs more arm folded than
            # that. Measured failure to pick the cube up at all, by spawn distance from
            # the base (4096 episodes, 2026-10-07):
            #
            #   0.30-0.35 m   1.0%      0.45-0.50 m  36.7%
            #   0.35-0.40 m   1.9%      0.50-0.56 m  75.6%
            #   0.40-0.45 m   9.6%
            #
            # Cube size and yaw had no effect on this (20-22% across every bin) and
            # removing the observation noise barely helped (75.6% -> 63.9% at the far
            # end), so it is reach, not perception. The old box's far corner
            # (0.50, 0.25) sat at 0.559 m, inside the kinematic bound but well past
            # what the policy can actually grasp from. The new corner is 0.457 m.
            #
            # The near edge and the lateral extent are limited by the camera instead:
            # at x = 0.30 the 85.7 deg FOV sees |y| < 0.278, and the cube's own half
            # width is 0.021, so |y| <= 0.18 now has 7.7 cm of margin where the old
            # +-0.25 had 7 mm.
            "pose_range": {
                "x": (-0.10, 0.02),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        # "prestartup" is this version's pre-simulation mode (the function's docstring
        # still calls it "usd"); it runs before sim.reset() parses the stage. A wrong
        # mode name here is silently ignored rather than raising.
        mode="prestartup",
        params={
            "scale_range": OBJECT_SCALE_RANGE,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Level 0 is the default pose every episode; the arm_init_level curriculum raises
    # each environment's level as it proves it can still hold the cube. Switched on at
    # full strength instead, this was the most damaging change in the 2026-10-06
    # ablation -- final lifting_object 0.232 against a 12.754 baseline.
    reset_robot_joints = EventTerm(
        func=mdp.reset_arm_joints_by_level,
        mode="reset",
        params={
            "max_range": ARM_INIT_MAX_RANGE,
            "num_levels": ARM_INIT_NUM_LEVELS,
            # arm joints only: the fingers are prismatic with 0.044 m of travel, so an
            # offset in radians would just pin them to a limit
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint.*"]),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
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
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


SMOOTHNESS_PENALTY_ONSET = 30000
"""When the action_rate / joint_vel penalties jump from 1e-4 to 1e-1, in env steps.

Counted in ``env.common_step_counter``, which advances once per environment step
regardless of ``num_envs``, so with ``num_steps_per_env = 24`` the ramp lands at
iteration ``SMOOTHNESS_PENALTY_ONSET / 24``.

The original 10000 puts it at iteration 417, and that is a cliff. In the 2026-10-06
ablation (config/sandbox/SUMMARY.md) every configuration that had started lifting by
iteration 399 went on to converge and every one that had not was crushed -- the action
noise std collapsed and the policy settled into reaching without ever grasping, with
nothing in between.

It is not only a property of hard tasks. On 2026-10-07 the same seed, with the cube's
spawn box *narrowed* so every position was within comfortable reach, drew an
exploration path that reached iteration 399 at lift 0.12 instead of the previous run's
0.28, hit the ramp, and never recovered over the remaining 1600 iterations. A strictly
easier task failed where a harder one had succeeded, on luck alone.

30000 puts the ramp at iteration 1250, by which point lifting is established with
margin. It costs convergence speed -- the ablation measured 8.95 lift at 1000
iterations with the ramp at 10000 against 6.83 at 1000 and 8.68 at 2500 with it at
30000 -- but that is roughly ten minutes against losing a whole run.
"""


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate",
            "weight": -1e-1,
            "num_steps": SMOOTHNESS_PENALTY_ONSET,
        },
    )

    arm_init_level = CurrTerm(func=mdp.arm_init_level)

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel",
            "weight": -1e-1,
            "num_steps": SMOOTHNESS_PENALTY_ONSET,
        },
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    # replicate_physics must be off for randomize_object_scale (Isaac Lab raises if it is
    # not). Measured cost of turning it off: scene build 4 s -> 48 s at 2048 envs, and
    # host RAM about 11.8 GB per 1024 envs. Step throughput is unaffected.
    #
    # Hence 2048 rather than the usual 4096: 4096 needs ~46 GB of the machine's 62 GB and
    # was OOM-killed with other work running. 2048 peaks at 24 GB.
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
