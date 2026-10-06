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
            # x/y are already saturated: the far corner (0.50, 0.25) sits 0.559 m from
            # the base against ~0.60 m of reach at table height, and the near corner
            # (0.30, 0.25) leaves 7 mm inside the camera's 85.7 deg FOV. yaw is the one
            # axis with room. Measurements in doc/design/student_distillation_design.md.
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.25, 0.25),
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

    # Deliberately no initial arm pose randomisation: a +-0.1 rad joint offset was the
    # most damaging change in the 2026-10-06 ablation, taking final lifting_object from
    # 8.952 to 0.121. See config/sandbox/SUMMARY.md before reintroducing it.


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


SMOOTHNESS_PENALTY_ONSET = 10000
"""When the action_rate / joint_vel penalties jump from 1e-4 to 1e-1, in env steps.

Counted in ``env.common_step_counter``, which advances once per environment step
regardless of ``num_envs``, so with ``num_steps_per_env = 24`` this lands at iteration
417 whatever the batch size.

This is a cliff, and worth knowing about before adding difficulty. In the 2026-10-06
ablation (config/sandbox/SUMMARY.md) every configuration that had started lifting by
iteration 399 went on to converge, and every one that had not was crushed by the ramp
and never recovered -- the action noise std collapsed to 0.07 and the policy settled
into reaching without ever grasping. There was nothing in between. Delaying the ramp
to 30000 keeps exploration alive but does not by itself make a too-hard task learnable.

Kept at the original 10000 because the delay only slowed things down once the real
blocker (initial arm pose randomisation) was removed: 8.95 lift at 1000 iterations
here, against 6.83 at 1000 and 8.68 at 2500 with the ramp at 30000.
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
