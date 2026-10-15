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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

ARM_INIT_LEVEL_ATTR = "arm_init_levels"
"""Where the per-environment difficulty level lives.

Held on the env rather than inside either manager because the reset event and the
curriculum both need it, and they run in different managers: the curriculum raises or
lowers a level, and the reset event that fires immediately afterwards spawns the arm at
whatever level that env now holds.
"""


def arm_init_levels(env: ManagerBasedEnv) -> torch.Tensor:
    """The per-environment level tensor, created at zero on first use."""
    levels = getattr(env, ARM_INIT_LEVEL_ATTR, None)
    if levels is None or levels.shape[0] != env.num_envs:
        levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, ARM_INIT_LEVEL_ATTR, levels)
    return levels


def reset_arm_joints_by_level(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_range: float,
    num_levels: int,
    asset_cfg: SceneEntityCfg,
    fixed_level: int | None = None,
) -> None:
    """Offset the arm joints by an amount set by each environment's difficulty level.

    Level 0 starts every episode from the default pose; level ``num_levels`` offsets each
    joint by up to ``+-max_range``. Isaac Lab's ``reset_joints_by_offset`` takes one
    range for the whole batch, which cannot express a per-environment curriculum, hence
    this variant.

    ``fixed_level`` pins every environment to one level and ignores the curriculum --
    used by the play config so that what is on screen is the difficulty the policy was
    finally trained at, rather than the level-0 default the curriculum would restart
    from outside training.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    ids = env_ids[:, None] if joint_ids != slice(None) else env_ids

    if fixed_level is None:
        level = arm_init_levels(env)[env_ids].float()
    else:
        level = torch.full((len(env_ids),), float(fixed_level), device=env.device)
    scale = (max_range * level / num_levels).unsqueeze(-1)

    joint_pos = asset.data.default_joint_pos[ids, joint_ids].clone()
    joint_pos += scale * (torch.rand_like(joint_pos) * 2.0 - 1.0)
    limits = asset.data.soft_joint_pos_limits[ids, joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids
    )


##
# Arm-Lift: object catalog, per-episode physics, and stable resting poses.
##

OBJECT_CATALOG_ATTR = "object_catalog"
"""Where the per-environment object description lives.

On the env for the same reason the arm-init levels are: the startup event that builds it,
the reset event that places the object, the observation terms that report it and the
reward that gates on its height all need it, and they live in four different managers.
"""

TABLE_TOP_Z = -0.003
"""World z of the table's *collision* surface, which is not where its visual surface is.

Measured by settling primitives of known half-height on it: every one came to rest exactly
3.0 mm below half its height, across sizes from 4.2 to 8.4 cm and with ``rest_offset``
forced to 0.0, -0.003 and +0.010 -- so this is the collider, not a contact offset. (The
DexCube the old cube task used sat 1.2 mm higher instead, because its collider is slightly
larger than its visual; no catalog entry here is a mesh, so that case no longer arises.)

Objects are placed 2 mm above their computed rest height rather than exactly on it, so an
error here costs a 2 mm drop instead of an interpenetration the solver has to push out of.
"""

REST_PLACEMENT_MARGIN = 0.002


class ObjectCatalogBuffers:
    """Per-environment object facts, resolved once at startup.

    ``half_extent``       (N, 3)     object-frame half-extents -- the privileged ``e^dim``
    ``category_onehot``   (N, C)     which primitive family
    ``rest_quat``         (N, P, 4)  stable attitudes, padded to P
    ``rest_height``       (N, P)     centre height above the table for each of them
    ``num_rest``          (N,)       how many of the P slots are real
    ``friction``          (N,)       static friction, resampled every episode
    ``rest_pose_height``  (N,)       the height of the pose this episode actually drew
    ``rest_pose_slot``    (N,)       which of the P attitudes that was -- for analysis only
    """

    def __init__(self, env: ManagerBasedEnv, catalog, env_to_entry: torch.Tensor) -> None:
        from ..object_catalog import CATEGORIES, MAX_REST_POSES

        device = env.device
        num_envs = env.num_envs
        self.entry_index = env_to_entry

        self.half_extent = torch.zeros(num_envs, 3, device=device)
        self.category_onehot = torch.zeros(num_envs, len(CATEGORIES), device=device)
        self.rest_quat = torch.zeros(num_envs, MAX_REST_POSES, 4, device=device)
        self.rest_height = torch.zeros(num_envs, MAX_REST_POSES, device=device)
        self.num_rest = torch.ones(num_envs, dtype=torch.long, device=device)
        self.nominal_mass = torch.zeros(num_envs, device=device)

        for env_id in range(num_envs):
            entry = catalog[int(env_to_entry[env_id])]
            self.half_extent[env_id] = torch.tensor(entry.half_extent, device=device)
            self.category_onehot[env_id, entry.category_index] = 1.0
            self.nominal_mass[env_id] = entry.nominal_mass
            poses = entry.rest_poses
            self.num_rest[env_id] = len(poses)
            for slot, (quat, height) in enumerate(poses):
                self.rest_quat[env_id, slot] = torch.tensor(quat, device=device)
                self.rest_height[env_id, slot] = height
            # Pad the unused slots with the first pose so an out-of-range index, if one
            # ever slipped through, would place the object legally rather than at the
            # origin with an all-zero quaternion.
            for slot in range(len(poses), MAX_REST_POSES):
                self.rest_quat[env_id, slot] = self.rest_quat[env_id, 0]
                self.rest_height[env_id, slot] = self.rest_height[env_id, 0]

        self.friction = torch.full((num_envs,), 0.8, device=device)
        self.rest_pose_height = self.rest_height[:, 0].clone()
        self.rest_pose_slot = torch.zeros(num_envs, dtype=torch.long, device=device)


def object_catalog(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> ObjectCatalogBuffers:
    """The buffers, built on first use.

    Built lazily rather than from a "startup" event because the observation manager is
    constructed first: ``ManagerBasedEnv.__init__`` calls ``load_managers()`` before it
    applies startup events, and the observation manager probes every term once while
    resolving its dimensions, so a startup event would always be too late. ``sim.reset()``
    has already run by that point, so the PhysX views this needs are live.

    Work out which catalog entry each environment was given, and cache its description.

    ``MultiAssetSpawnerCfg`` assigns ``proto_prim_paths[i % len(assets_cfg)]`` to the i-th
    prim path it finds (``sim/spawners/wrappers/wrappers.py:110``). The tempting shortcut is
    to assume i is the environment number, but the paths come from a USD regex match whose
    order is not promised to be numeric -- ``env_10`` may well sort before ``env_2``. So the
    same lookup is redone here against the same path list, and the environment number is
    parsed out of each path rather than assumed.

    The mapping is then checked against the masses PhysX actually reports. Getting this
    wrong would not crash anything; it would quietly train the policy on another object's
    dimensions, which is the one failure mode in this file worth paying for a check.
    """
    cached = getattr(env, OBJECT_CATALOG_ATTR, None)
    if cached is not None:
        return cached

    import re

    import isaaclab.sim as sim_utils

    from ..object_catalog import build_catalog

    catalog = build_catalog()
    asset = env.scene[asset_cfg.name]

    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if len(prim_paths) != env.num_envs:
        raise RuntimeError(
            f"Expected {env.num_envs} object prims for '{asset_cfg.name}', found {len(prim_paths)}."
        )

    env_to_entry = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for index, path in enumerate(prim_paths):
        match = re.search(r"/env_(\d+)/", path)
        if match is None:
            raise RuntimeError(f"Could not read an environment index out of prim path '{path}'.")
        env_to_entry[int(match.group(1))] = index % len(catalog)

    buffers = ObjectCatalogBuffers(env, catalog, env_to_entry)
    setattr(env, OBJECT_CATALOG_ATTR, buffers)

    measured = asset.root_physx_view.get_masses().to(env.device).reshape(env.num_envs)
    error = (measured - buffers.nominal_mass).abs() / buffers.nominal_mass.clamp(min=1e-9)
    if error.max() > 0.02:
        worst = int(error.argmax())
        raise RuntimeError(
            "Catalog-to-environment mapping is wrong: environment "
            f"{worst} was matched to entry {int(env_to_entry[worst])} "
            f"(expected mass {buffers.nominal_mass[worst]:.6f} kg) but PhysX reports "
            f"{measured[worst]:.6f} kg."
        )

    return buffers


def randomize_object_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    density_range: tuple[float, float],
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Resample the object's density multiplier and friction for the coming episode.

    Isaac Lab ships ``randomize_rigid_body_mass`` and ``randomize_rigid_body_material``
    which between them do the same work, but neither hands back what it drew, and both
    quantities have to appear in the privileged observation. Sampling here keeps the value
    the policy is told and the value the simulator uses provably the same tensor.

    Density is a multiplier on the catalog's nominal, as in the RMA paper (Tab. 2), so a
    small object stays light and a large one heavy; it is the material that varies, not the
    size. Friction is set static-equal-to-dynamic: a separate dynamic coefficient would be
    a second hidden variable with no separate observable consequence here.
    """
    buffers = object_catalog(env)
    asset = env.scene[asset_cfg.name]
    cpu_ids = env_ids.cpu()

    multiplier = torch.rand(len(env_ids), device=env.device)
    multiplier = density_range[0] + multiplier * (density_range[1] - density_range[0])
    masses = asset.root_physx_view.get_masses()
    masses[cpu_ids, 0] = (buffers.nominal_mass[env_ids] * multiplier).cpu()
    asset.root_physx_view.set_masses(masses, cpu_ids)

    friction = torch.rand(len(env_ids), device=env.device)
    friction = friction_range[0] + friction * (friction_range[1] - friction_range[0])
    buffers.friction[env_ids] = friction
    materials = asset.root_physx_view.get_material_properties()
    # (num_envs, num_shapes, 3) = static friction, dynamic friction, restitution.
    materials[cpu_ids, :, 0] = friction.cpu().unsqueeze(-1)
    materials[cpu_ids, :, 1] = friction.cpu().unsqueeze(-1)
    asset.root_physx_view.set_material_properties(materials, cpu_ids)


def reset_object_stable_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Place the object in one of its stable resting poses, at a random spot and yaw.

    Isaac Lab's ``reset_root_state_uniform`` cannot express this. Sampling roll and pitch
    from a continuous range drops the object in an attitude it has no reason to stay in, so
    it tumbles through the first tenth of a second of every episode; and the height it
    should be placed at depends on which attitude was drawn, which a fixed ``z`` range
    cannot follow. A lying cylinder rests at ``r``, the same cylinder upright rests at
    ``h/2``, and a cone on its side rests at ``(h/2) sin(atan(r/h))``.

    The pose is drawn uniformly over the entry's stable attitudes, then spun by a uniform
    yaw about the world vertical -- which keeps it stable, because the vertical is what the
    attitude was chosen to resolve.
    """
    from isaaclab.utils.math import quat_mul

    buffers = object_catalog(env)
    asset = env.scene[asset_cfg.name]
    num = len(env_ids)

    slot = (torch.rand(num, device=env.device) * buffers.num_rest[env_ids]).long()
    slot = torch.minimum(slot, buffers.num_rest[env_ids] - 1)
    rest_quat = buffers.rest_quat[env_ids, slot]
    rest_height = buffers.rest_height[env_ids, slot]
    buffers.rest_pose_height[env_ids] = rest_height
    buffers.rest_pose_slot[env_ids] = slot

    def _draw(key: str) -> torch.Tensor:
        low, high = pose_range.get(key, (0.0, 0.0))
        return low + torch.rand(num, device=env.device) * (high - low)

    yaw = _draw("yaw")
    half = 0.5 * yaw
    yaw_quat = torch.zeros(num, 4, device=env.device)
    yaw_quat[:, 0] = torch.cos(half)
    yaw_quat[:, 3] = torch.sin(half)

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env.scene.env_origins[env_ids] + asset.data.default_root_state[env_ids, :3]
    root_state[:, 0] += _draw("x")
    root_state[:, 1] += _draw("y")
    root_state[:, 2] = (
        env.scene.env_origins[env_ids, 2] + TABLE_TOP_Z + rest_height + REST_PLACEMENT_MARGIN
    )
    root_state[:, 3:7] = quat_mul(yaw_quat, rest_quat)
    root_state[:, 7:] = 0.0

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
