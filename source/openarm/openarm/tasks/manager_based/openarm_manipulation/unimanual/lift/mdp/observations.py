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


##
# Arm-Lift: observable derived quantities, and the privileged vector.
##

# Every privileged term is scaled into roughly [-1, 1] here rather than by rsl-rl's
# empirical normalisation. The ranges are ours -- we set the randomisation -- so a fixed
# divisor is exact, whereas running statistics are a moving target over exactly the early
# iterations this task has already been shown to be fragile in (see
# SMOOTHNESS_PENALTY_ONSET in lift_env_cfg.py). Without scaling of some kind the encoder's
# first layer is dominated by contact force, which is tens of newtons while a half-extent
# is hundredths of a metre.
EXTENT_SCALE = 0.05
MASS_SCALE = 0.2
FRICTION_CENTRE = 0.8
FRICTION_SCALE = 0.3
CONTACT_FORCE_SCALE = 4.0


def tcp_to_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object position relative to the TCP, in the robot's root frame.

    Derivable from terms the policy already has, but only through the forward kinematics of
    a 7-DoF arm. Handing it over costs three floats and saves the network learning that;
    the RMA paper does the same (section 4, the derived goal terms).
    """
    from isaaclab.utils.math import quat_apply_inverse

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - tcp_w)


def object_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal position relative to the object, in the robot's root frame."""
    from isaaclab.utils.math import quat_apply_inverse

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    goal_b = env.command_manager.get_command(command_name)[:, :3]
    object_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    return goal_b - object_b


def object_extent(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Half-extents along the object's own axes -- the RMA paper's ``e^dim``.

    The object frame, not the resting frame: which way up it currently is comes from
    ``object_rot_6d``, and pre-combining the two here would only duplicate it.
    """
    from .events import object_catalog

    return object_catalog(env).half_extent / EXTENT_SCALE


def object_category(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot over the primitive families.

    This reaches the encoder as a one-hot and becomes an embedding table there: the first
    layer is a Linear, so its columns against these five slots *are* a learned 32-vector per
    category, identical to ``nn.Embedding(5, 32)``. Written as a one-hot on this side so the
    observation stays float, and as a bias-free Linear on the model side so the table can be
    dumped and looked at.
    """
    from .events import object_catalog

    return object_catalog(env).category_onehot


def object_mass(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Mass as the simulator currently holds it, after the per-episode density draw."""
    obj: RigidObject = env.scene[object_cfg.name]
    masses = obj.root_physx_view.get_masses().to(env.device).reshape(env.num_envs, 1)
    return masses / MASS_SCALE


def object_friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Static friction drawn for this episode.

    Read from the buffer the randomisation event wrote rather than back out of PhysX, so the
    number the policy is told cannot drift from the number the solver is using.
    """
    from .events import object_catalog

    friction = object_catalog(env).friction.unsqueeze(-1)
    return (friction - FRICTION_CENTRE) / FRICTION_SCALE


def object_rot_6d(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object attitude in the robot's root frame, as the first two columns of its matrix.

    Not Euler angles or a rotation vector. Yaw is randomised over the full circle every
    episode, so a three-number parameterisation would cross its own wrap-around constantly:
    one degree of physical rotation, a jump of 2*pi in the input. Six numbers is the smallest
    continuous description of SO(3), and the third column is the cross product of these two,
    so nothing is lost.
    """
    from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    relative = quat_mul(quat_conjugate(robot.data.root_quat_w), obj.data.root_quat_w)
    matrix = matrix_from_quat(relative)
    return torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=-1)


def finger_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_contact"),
) -> torch.Tensor:
    """Magnitude of the net contact force on each finger -- the RMA paper's ``e^impl``.

    Partly inferable from proprioception: with an implicit actuator the commanded force is
    ``stiffness * (target - position)`` and both terms are observed. What is not inferable is
    the part that matters -- whether contact is happening at all, and whether it is happening
    on both fingers. An off-centre grasp shows up here as an asymmetry and nowhere else.

    ``log1p`` because contact force is heavy-tailed: it spikes on impact, and a linear scale
    chosen to keep those spikes inside [-1, 1] would flatten the low-force region where the
    interesting distinction between "touching" and "gripping" lives.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.log1p(torch.norm(forces, dim=-1)) / CONTACT_FORCE_SCALE
