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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def depth_image_flat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    data_type: str = "distance_to_image_plane",
    near: float = 0.1,
    far: float = 1.2,
) -> torch.Tensor:
    """Flattened depth image, clipped to the camera range and scaled to [0, 1].

    The observation manager concatenates terms with ``torch.cat`` and does not flatten, so the
    image has to be returned as ``(num_envs, height * width)`` to sit alongside the proprioceptive
    terms in the same group. The student model reshapes it back (see
    ``openarm.assets.models.depth_student``).

    Rays that hit nothing come back as ``inf``; they are mapped to ``far`` so the stored
    observation stays finite.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    image = sensor.data.output[data_type]
    # (N, H, W, 1) or (N, H, W) depending on the sensor
    image = image.squeeze(-1) if image.ndim == 4 else image
    image = torch.nan_to_num(image, nan=far, posinf=far, neginf=near)
    image = image.clamp(near, far)
    return ((image - near) / (far - near)).flatten(start_dim=1)
