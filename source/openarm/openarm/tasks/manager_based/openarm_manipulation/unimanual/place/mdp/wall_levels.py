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

"""Wall height by level, after the terrain-level curriculum in Isaac Lab's locomotion tasks.

Each environment holds a level; the level sets how far the wall ring stands above the box
floor. Success moves an environment up a level, failing to get the object into the box at
all moves it down, and an environment that succeeds at the top level is sent to a random
one -- ``TerrainImporter.update_env_origins`` does the same, so the easy levels keep being
seen. Nothing is rebuilt: the ring is built once at full height and only moved, sunk into
the table by however much the level does not need. A kinematic body against the static
table makes no contacts.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

WALL_LEVEL_ATTR = "wall_levels"


def wall_levels(env: ManagerBasedEnv) -> torch.Tensor:
    """The per-environment level tensor, created at zero on first use."""
    levels = getattr(env, WALL_LEVEL_ATTR, None)
    if levels is None or levels.shape[0] != env.num_envs:
        levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, WALL_LEVEL_ATTR, levels)
    return levels


def wall_heights(env: ManagerBasedEnv) -> torch.Tensor:
    """Each environment's wall height above the box floor, in metres."""
    table = torch.tensor(env.cfg.wall_levels, device=env.device)
    return table[wall_levels(env)]


def reset_bin_walls(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    fixed_level: int | None = None,
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    walls_cfg: SceneEntityCfg = SceneEntityCfg("bin_walls"),
) -> None:
    """Stand the wall ring on the box floor at the level's height. Must follow reset_bin.

    ``fixed_level`` pins every environment to one level, for play and evaluation; None or a
    negative value follows the curriculum.
    """
    if fixed_level is not None and fixed_level >= 0:
        wall_levels(env)[env_ids] = fixed_level
    built = env.cfg.bin_geometry.wall_height
    pose = env.scene[bin_cfg.name].data.root_pose_w[env_ids].clone()
    pose[:, 2] -= built - wall_heights(env)[env_ids]
    env.scene[walls_cfg.name].write_root_pose_to_sim(pose, env_ids=env_ids)


def wall_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
    """Curriculum term: up on success, down on not getting the object into the box at all.

    Runs before the reset events, so the new level is the one ``reset_bin_walls`` builds to.
    """
    from .rewards import object_in_bin, place_success

    levels = wall_levels(env)
    num_levels = len(env.cfg.wall_levels)
    up = place_success(env)[env_ids]
    down = ~object_in_bin(env)[env_ids]
    new = levels[env_ids] + up.long() - down.long()
    solved = new >= num_levels
    new = torch.where(solved, torch.randint_like(new, num_levels), new.clamp(min=0))
    levels[env_ids] = new
    return float(levels.float().mean())
