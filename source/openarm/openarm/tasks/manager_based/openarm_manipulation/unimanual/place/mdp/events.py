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

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ...lift.mdp.events import TABLE_TOP_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

BIN_PLACEMENT_TRIES = 10


def reset_bin_near_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> None:
    """Put the bin ``distance_range`` from the object, in a random direction, inside the box.

    Must run after the object has been placed: it reads the object's new position, which
    ``write_root_pose_to_sim`` has already written into the data buffers.

    ManiSkill's layout cannot be copied over. Measured from the robot base it puts the
    sphere at 0.52-0.57 m and the bin at 0.62-0.72 m, and OpenArm already fails to pick
    the cube up 75.6% of the time beyond 0.50 m. So both live in the box the lift task
    measured as reachable and in camera view, and only the separation is taken from
    ManiSkill (0.05-0.28 m there).

    Directions are drawn ``BIN_PLACEMENT_TRIES`` times and the first that lands in the box is
    kept. The box is 0.36 m wide in y, so straight across always fits; an environment that
    misses on every try gets that instead.
    """
    obj = env.scene[object_cfg.name]
    bin_asset = env.scene[bin_cfg.name]
    num = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    obj_xy = obj.data.root_pos_w[env_ids, :2] - origins[:, :2]

    # Fallback: straight across in y, to the side with more room.
    side = torch.where(obj_xy[:, 1] >= 0.0, -1.0, 1.0)
    bin_xy = obj_xy.clone()
    bin_xy[:, 1] += side * 0.5 * (distance_range[0] + distance_range[1])
    found = torch.zeros(num, dtype=torch.bool, device=env.device)

    for _ in range(BIN_PLACEMENT_TRIES):
        theta = torch.rand(num, device=env.device) * 2.0 * math.pi
        d = distance_range[0] + torch.rand(num, device=env.device) * (distance_range[1] - distance_range[0])
        candidate = obj_xy + d.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        inside = (
            (candidate[:, 0] >= x_range[0])
            & (candidate[:, 0] <= x_range[1])
            & (candidate[:, 1] >= y_range[0])
            & (candidate[:, 1] <= y_range[1])
        )
        take = inside & ~found
        bin_xy[take] = candidate[take]
        found |= inside

    pose = torch.zeros(num, 7, device=env.device)
    pose[:, :2] = origins[:, :2] + bin_xy
    pose[:, 2] = origins[:, 2] + TABLE_TOP_Z + env.cfg.bin_geometry.floor_half_thickness
    pose[:, 3] = 1.0
    bin_asset.write_root_pose_to_sim(pose, env_ids=env_ids)
