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
from ..bin_asset import BinCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

IN_BIN_XY_TOL = 0.005
IN_BIN_Z_TOL = 0.005
"""ManiSkill's 5 mm on both. In xy it is measured from the square the object's centre can
reach on the floor, so it means "in the pocket" whatever the pocket's play -- and in
ManiSkill's own bin, where the play is zero, it is exactly ManiSkill's 5 mm from the centre."""

GRASP_MIN_FORCE = 0.5
"""Newtons on *each* finger, as ManiSkill's ``is_grasping``. Its 85 deg direction check is
left out; with the contact filtered to the object, a one-sided push already fails."""

STATIC_LIN_VEL = 0.01
STATIC_ANG_VEL = 0.5
"""ManiSkill's ``is_static(lin_thresh=1e-2, ang_thresh=0.5)``, but applied to the
*horizontal* angular velocity only -- see ``horizontal_ang_vel``."""

FINGER_OPEN = 0.044


def bin_geometry(env: ManagerBasedRLEnv) -> BinCfg:
    return env.cfg.bin_geometry


def place_target_w(
    env: ManagerBasedRLEnv, bin_cfg: SceneEntityCfg = SceneEntityCfg("bin")
) -> torch.Tensor:
    """World position the object's centre has when it rests on the bin floor.

    The floor top plus the object's resting height, so it stays right if the catalog ever
    holds something other than one sphere.
    """
    bin_pos = env.scene[bin_cfg.name].data.root_pos_w
    target = bin_pos.clone()
    target[:, 2] += bin_geometry(env).floor_half_thickness + object_catalog(env).rest_pose_height
    return target


def object_in_bin(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """The object is resting on the bin floor, within the pocket."""
    offset = env.scene[object_cfg.name].data.root_pos_w - place_target_w(env)
    # How far the centre can move from the middle before the object touches a wall.
    half_width = object_catalog(env).half_extent[:, :2].max(dim=1).values
    play = (bin_geometry(env).inner_half - half_width).clamp(min=0.0)
    outside = (offset[:, :2].abs() - play.unsqueeze(-1)).clamp(min=0.0)
    return (torch.norm(outside, dim=1) <= IN_BIN_XY_TOL) & (offset[:, 2].abs() <= IN_BIN_Z_TOL)


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
    return (offset[:, :2].abs() <= bin_geometry(env).outer_half).all(dim=1)


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


def object_on_target(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """In the bin, and within ``env.cfg.place_tolerance`` of its centre if one is set.

    Unset, anywhere in the pocket will do, which is all a pocket as narrow as the object
    allows anyway. In a bin wide enough for the hand to go in, it is what makes placement
    precise: a sphere dropped from above the rim tends to roll off the mark.
    """
    in_bin = object_in_bin(env)
    tolerance = getattr(env.cfg, "place_tolerance", None)
    if tolerance is None:
        return in_bin
    offset = env.scene[object_cfg.name].data.root_pos_w[:, :2] - place_target_w(env)[:, :2]
    return in_bin & (torch.norm(offset, dim=1) <= tolerance)


def placement_valid(env: ManagerBasedRLEnv) -> torch.Tensor:
    """See ``release.placement_valid``. Imported late: release.py builds on this module."""
    from .release import placement_valid as _placement_valid

    return _placement_valid(env)


def place_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ManiSkill's success: on target, at rest (spin about the vertical allowed), and let go
    of -- by hand, where the task requires it (``placement_valid``).

    ``env.cfg.success_requires_rest = False`` drops the rest check: on target and let go of
    is enough, rolling or not.
    """
    success = object_on_target(env) & ~is_grasped(env) & placement_valid(env)
    if getattr(env.cfg, "success_requires_rest", True):
        success &= object_is_static(env)
    return success


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
    # placement_valid: a sphere dropped in is over the bin too, and must not be paid for it.
    over_bin = object_over_bin(env) & placement_valid(env)
    return (lifted | over_bin) * (1 - torch.tanh(distance / std))


RIM_CLEARANCE = 0.02
"""How far above the rim the object's lowest point is aimed when carried over it."""


def place_tracking_over_rim(
    env: ManagerBasedRLEnv,
    std: float,
    margin: float,
    clearance: float = RIM_CLEARANCE,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """``place_tracking`` for a bin too deep to aim straight at the floor.

    With walls twice the sphere's height, the straight line from a lifted sphere to the floor
    target runs through a wall, and shaping along it pushes the sphere into the wall's side.
    So the target is instead a point ``clearance`` above the rim, and once the object is over
    the bin the term pays in full -- full rather than switching to the floor target, because
    the floor is further from a sphere at rim height than the waypoint was, and the reward
    would drop at the moment of arriving. ``place_tracking_fine`` keeps aiming at the floor
    and does the pulling down.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    waypoint = env.scene[bin_cfg.name].data.root_pos_w.clone()
    rest_height = object_catalog(env).rest_pose_height
    waypoint[:, 2] += bin_geometry(env).rim_height + rest_height + clearance
    distance = torch.norm(waypoint - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > lift_threshold(env, margin)
    approach = lifted * (1 - torch.tanh(distance / std))
    over_bin = object_over_bin(env) & placement_valid(env)
    return torch.where(over_bin, torch.ones_like(approach), approach)


def object_in_bin_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pays for the object being in the bin, held or -- where the task requires it -- put
    down by hand rather than dropped in (``placement_valid``).

    Weighted above ``lifting_object``: lowering the object into the bin gives up the lift
    reward, so this has to be worth more for lowering to pay.
    """
    return (object_in_bin(env) & placement_valid(env)).float()


def released_in_bin(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
) -> torch.Tensor:
    """On target, not grasped, scaled by how far the gripper is open.

    ManiSkill pays the opening fraction while the object is still held and a flat 16 once it
    is not. Here only the released case pays, and the opening keeps it from being collected
    by fingers that merely went slack against the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    opening = (robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=1) / FINGER_OPEN).clamp(0.0, 1.0)
    return (object_on_target(env) & ~is_grasped(env) & placement_valid(env)) * opening


def static_in_bin(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """On target and settling: ManiSkill's ``1 - tanh(10 v + w)``, with ``w`` horizontal only
    for the same reason as ``object_is_static``."""
    obj: RigidObject = env.scene[object_cfg.name]
    v = torch.norm(obj.data.root_lin_vel_w, dim=1)
    w = horizontal_ang_vel(obj)
    return (object_on_target(env) & placement_valid(env)) * (1 - torch.tanh(10.0 * v + w))


def place_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return place_success(env).float()


##
# Staged terms for the hand-sized box
##


def object_inside_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """The object's centre is inside the box: within the inner walls and below the rim."""
    offset = env.scene[object_cfg.name].data.root_pos_w - env.scene[bin_cfg.name].data.root_pos_w
    geometry = bin_geometry(env)
    return (offset[:, :2].abs() <= geometry.inner_half).all(dim=1) & (offset[:, 2] <= geometry.rim_height)


def held_or_placed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripped, or let go of low enough to count as put down by hand.

    The box rewards key on this rather than on "lifted", for two reasons. Letting go must not
    switch the holding rewards off -- if it does, releasing costs everything holding paid,
    and at model_1999 of the box tasks (2026-10-22/23) the policy never let go: 76-87% of
    episodes ended gripping the object on the floor, centred to 0.6 cm. And an object left
    on top of a wall is lifted without being held: the cube task paid it, and 19% of
    episodes ended with the cube parked on the rim.
    """
    from .release import release_tracker

    tracker = release_tracker(env)
    tracker.update(env)
    return is_grasped(env) | tracker.placed | tracker.releasing_low


def carrying(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """Held or put down by hand, and either lifted or inside the box.

    "Inside the box" matters at the bottom: the box floor puts a resting object only 5 mm
    above where it rests on the table, inside the 2 cm lift margin, so plain "lifted" would
    switch off just before the object reaches the floor and take the lift and carry rewards
    with it.
    """
    obj: RigidObject = env.scene["object"]
    lifted = obj.data.root_pos_w[:, 2] > lift_threshold(env, margin)
    return held_or_placed(env) & (lifted | object_inside_box(env))


def carrying_reward(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    return carrying(env, margin).float()


def carry_to_box(
    env: ManagerBasedRLEnv,
    std: float,
    margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """While carrying, closeness to the box centre in the horizontal plane only.

    No waypoint over the rim: an object pushed against the outside of a wall is still 12 cm
    from the centre, so the wall caps this below what going over it pays, and lifting higher
    on the way costs nothing.
    """
    offset = env.scene[object_cfg.name].data.root_pos_w[:, :2] - env.scene[bin_cfg.name].data.root_pos_w[:, :2]
    return carrying(env, margin) * (1 - torch.tanh(torch.norm(offset, dim=1) / std))


def held_over_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """Held or put down by hand, with the object's centre within the inner walls seen from
    above -- at any height."""
    offset = env.scene[object_cfg.name].data.root_pos_w - env.scene[bin_cfg.name].data.root_pos_w
    over = (offset[:, :2].abs() <= bin_geometry(env).inner_half).all(dim=1)
    return over & held_or_placed(env)


def lower_to_target(
    env: ManagerBasedRLEnv, std: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """While held over the box, closeness to the resting point at the floor's centre.

    Over, not inside: gated on "below the rim", this paid nothing to a sphere held high above
    the box, and the policy held it there -- 49 cm above the floor at model_1999 (2026-10-22).
    """
    distance = torch.norm(place_target_w(env) - env.scene[object_cfg.name].data.root_pos_w, dim=1)
    return held_over_box(env) * (1 - torch.tanh(distance / std))


def released_on_target(
    env: ManagerBasedRLEnv, radius: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Let go of by hand on the box floor: 1 at the centre, falling linearly to 0 at ``radius``.

    A bonus on top of the holding rewards, which a placed object keeps (``held_or_placed``),
    so letting go anywhere inside ``radius`` gains something and costs nothing. Graded rather
    than all-or-nothing at the 1 cm success tolerance, so an imprecise release still pays.
    """
    offset = env.scene[object_cfg.name].data.root_pos_w[:, :2] - place_target_w(env)[:, :2]
    closeness = (1 - torch.norm(offset, dim=1) / radius).clamp(min=0.0)
    return (object_in_bin(env) & ~is_grasped(env) & placement_valid(env)) * closeness
