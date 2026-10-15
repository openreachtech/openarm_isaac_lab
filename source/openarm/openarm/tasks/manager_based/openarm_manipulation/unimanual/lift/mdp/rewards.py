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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


##
# Arm-Lift: height gates measured from where the object actually rests.
##


def lift_threshold(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """World z the object has to clear, per environment.

    The cube task compared against a fixed 0.04 m, which only worked because there was one
    object. With a mixed catalog an absolute gate asks different amounts of work from
    different shapes -- an upright cylinder 5.5 cm tall rests at 2.75 cm and has 1.25 cm to
    climb, while a 3 cm sphere rests at 1.5 cm and has 2.5 cm to climb. The policy would
    learn to prefer tall objects and the per-shape success rates would not be comparable.

    Measuring from the resting height instead asks every object for the same ``margin``.
    """
    from .events import TABLE_TOP_Z, object_catalog

    return env.scene.env_origins[:, 2] + TABLE_TOP_Z + object_catalog(env).rest_pose_height + margin


def object_is_lifted_relative(
    env: ManagerBasedRLEnv,
    margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting the object ``margin`` above where it was resting."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] > lift_threshold(env, margin)).float()


def object_goal_distance_relative(
    env: ManagerBasedRLEnv,
    std: float,
    margin: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``object_goal_distance`` with the lift gate measured from the resting height."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > lift_threshold(env, margin)
    return lifted * (1 - torch.tanh(distance / std))
