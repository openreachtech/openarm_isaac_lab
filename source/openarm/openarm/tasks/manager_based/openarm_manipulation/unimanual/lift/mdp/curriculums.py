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
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.math import combine_frame_transforms

from .events import arm_init_levels

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_init_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    minimal_height: float = 0.04,
    goal_distance: float = 0.10,
    command_name: str = "object_pose",
) -> float:
    """Promote or demote each environment's initial-pose difficulty, one level at a time.

    Modelled on the terrain-level curriculum used for quadrupeds: the level is held per
    environment rather than globally, so at any moment the batch spans a range of
    difficulties and the easy environments keep supplying signal while the hard ones are
    still being learned. The logged mean level is then a single number that says how far
    the whole population has got, and a policy can be evaluated at a stated level.

    Starting the randomisation at full strength does not work -- at +-0.1 rad it was the
    most damaging change in the 2026-10-06 ablation, taking final ``lifting_object``
    from 12.754 to 0.232 -- but reaching the same range through a curriculum is fine,
    and on 2026-10-09 came out slightly ahead of no randomisation at all.

    Promotion asks for the task to have been *done*, not merely survived: the cube has
    to have finished within ``goal_distance`` of where it was asked to go. Promoting on
    "still holding it" alone let the level race to maximum by iteration 750, while
    placement was still unlearned at ``fine_grained`` 0.02, and one seed in two never
    recovered the skill -- it ended holding the cube reliably (``lifting_object`` 11.8)
    and putting it nowhere (``fine_grained`` 0.072, against 1.596 for the other seed).
    This mirrors the quadruped terrain curriculum, which promotes on distance travelled
    rather than on staying upright.

    Demotion stays on losing the cube, so an environment that holds on but places poorly
    keeps practising at its level rather than being pushed back down.

    Both run before ``scene.reset``, so the new level is the one the arm spawns at.
    """
    levels = arm_init_levels(env)
    obj, robot = env.scene["object"], env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], command[env_ids, :3]
    )
    held = obj.data.root_pos_w[env_ids, 2] > minimal_height
    placed = held & (torch.norm(des_pos_w - obj.data.root_pos_w[env_ids], dim=1) < goal_distance)

    step = torch.where(placed, 1, torch.where(held, 0, -1))
    num_levels = env.cfg.events.reset_robot_joints.params["num_levels"]
    levels[env_ids] = torch.clamp(levels[env_ids] + step, 0, num_levels)
    return float(levels.float().mean())
