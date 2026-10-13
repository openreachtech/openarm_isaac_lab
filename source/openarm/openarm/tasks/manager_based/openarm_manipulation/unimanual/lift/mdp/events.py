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
