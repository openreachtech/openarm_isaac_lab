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

"""Where the object is relative to the bin, and the rewards built on it.

The predicates (``object_in_bin``, ``is_grasped``, ``place_success``) are shared by the
rewards, the curriculum and the success metric, so all three agree on what "placed" means.
Thresholds follow ManiSkill's ``PlaceSphere-v1.evaluate``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ...lift.mdp.events import object_catalog
from ...lift.mdp.rewards import lift_threshold
from ..bin_asset import BIN_FLOOR_HALF_THICKNESS, BIN_OUTER_HALF

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

IN_BIN_XY_TOL = 0.005
IN_BIN_Z_TOL = 0.005
"""ManiSkill's 5 mm on both. The pocket is as wide as the sphere, so the xy check mostly
confirms that it is in the pocket rather than resting on a wall."""

GRASP_MIN_FORCE = 0.5
"""Newtons on *each* finger, as ManiSkill's ``is_grasping``. Its 85 deg direction check is
left out; with the contact filtered to the object, a one-sided push already fails."""

STATIC_LIN_VEL = 0.01
STATIC_ANG_VEL = 0.5
"""ManiSkill's ``is_static(lin_thresh=1e-2, ang_thresh=0.5)``, but applied to the
*horizontal* angular velocity only -- see ``horizontal_ang_vel``."""

FINGER_OPEN = 0.044


def place_target_w(
    env: ManagerBasedRLEnv, bin_cfg: SceneEntityCfg = SceneEntityCfg("bin")
) -> torch.Tensor:
    """World position the object's centre has when it rests on the bin floor.

    The floor top plus the object's resting height, so it stays right if the catalog ever
    holds something other than one sphere.
    """
    bin_pos = env.scene[bin_cfg.name].data.root_pos_w
    target = bin_pos.clone()
    target[:, 2] += BIN_FLOOR_HALF_THICKNESS + object_catalog(env).rest_pose_height
    return target


def object_in_bin(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """The object is resting on the bin floor, within the pocket."""
    offset = env.scene[object_cfg.name].data.root_pos_w - place_target_w(env)
    return (torch.norm(offset[:, :2], dim=1) <= IN_BIN_XY_TOL) & (offset[:, 2].abs() <= IN_BIN_Z_TOL)


def object_over_bin(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """The object's centre is within the bin's outer square, seen from above.

    A sphere resting on the table cannot be here -- it would be inside a wall -- so this
    only ever holds while the object is carried over the bin or lowered into it.
    """
    offset = env.scene[object_cfg.name].data.root_pos_w - env.scene[bin_cfg.name].data.root_pos_w
    return (offset[:, :2].abs() <= BIN_OUTER_HALF).all(dim=1)


FINGER_OBJECT_SENSORS = ("left_finger_object_contact", "right_finger_object_contact")
"""One per finger. PhysX wants one filter match per sensor body, so a single sensor over
both fingers filtered to one object fails at startup ("expected 128, found 64")."""


def is_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Both fingers press on the object with at least ``GRASP_MIN_FORCE``."""
    grasped = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in FINGER_OBJECT_SENSORS:
        forces = env.scene.sensors[name].data.force_matrix_w  # (N, 1, 1, 3)
        grasped &= torch.norm(forces, dim=-1).sum(dim=(1, 2)) >= GRASP_MIN_FORCE
    return grasped


def horizontal_ang_vel(obj: RigidObject) -> torch.Tensor:
    """Angular speed about horizontal axes -- rolling -- with spin about the vertical left out.

    A sphere spinning in place about the vertical is still where it was put, and PhysX will
    not stop it: the contact with the floor is a single point, so sliding friction has no
    lever arm about the normal and there is no torsional friction unless a patch radius is
    set. Measured on model_1999 (6000 episodes), 10.5% of all episodes ended in the bin, let
    go of and untouched, with zero linear velocity and a median 1.1 rad/s about z -- 91% of
    all failures. Rolling, by contrast, does move the sphere, so that is what is checked.
    """
    return torch.norm(obj.data.root_ang_vel_w[:, :2], dim=1)


def object_is_static(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return (torch.norm(obj.data.root_lin_vel_w, dim=1) <= STATIC_LIN_VEL) & (
        horizontal_ang_vel(obj) <= STATIC_ANG_VEL
    )


def place_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ManiSkill's success: in the bin, at rest (spin about the vertical allowed), and let go of."""
    return object_in_bin(env) & object_is_static(env) & ~is_grasped(env)


##
# Reward terms
##


def place_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``object_goal_distance_relative`` with the goal on the bin floor.

    The lift task gates this on the object being lifted, but the goal here is barely above
    the resting height -- 5 mm, against a 2 cm lift margin -- so that gate would switch the
    reward off at the moment the object is lowered into the bin, and the policy would learn
    to hover. The gate is therefore "lifted *or* over the bin".
    """
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.norm(place_target_w(env) - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > lift_threshold(env, margin)
    return (lifted | object_over_bin(env)) * (1 - torch.tanh(distance / std))


def object_in_bin_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pays for the object being in the bin, held or not.

    Weighted above ``lifting_object``: lowering the object into the bin gives up the lift
    reward, so this has to be worth more for lowering to pay.
    """
    return object_in_bin(env).float()


def released_in_bin(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
) -> torch.Tensor:
    """In the bin, not grasped, scaled by how far the gripper is open.

    ManiSkill pays the opening fraction while the object is still held and a flat 16 once it
    is not. Here only the released case pays, and the opening keeps it from being collected
    by fingers that merely went slack against the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    opening = (robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=1) / FINGER_OPEN).clamp(0.0, 1.0)
    return (object_in_bin(env) & ~is_grasped(env)) * opening


def static_in_bin(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """In the bin and settling: ManiSkill's ``1 - tanh(10 v + w)``, with ``w`` horizontal only
    for the same reason as ``object_is_static``."""
    obj: RigidObject = env.scene[object_cfg.name]
    v = torch.norm(obj.data.root_lin_vel_w, dim=1)
    w = horizontal_ang_vel(obj)
    return object_in_bin(env) * (1 - torch.tanh(10.0 * v + w))


def place_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return place_success(env).float()
