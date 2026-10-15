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

"""The object catalog for ``Arm-Lift``: five primitive families, sampled parametrically.

Isaac Lab binds an environment's object at prim-creation time -- ``MultiAssetSpawnerCfg``
hands out ``assets_cfg[index % len(assets_cfg)]`` when ``random_choice`` is False
(``isaaclab/sim/spawners/wrappers/wrappers.py:110``). Shapes therefore cannot change at
reset, so "sample the dimensions continuously" has to mean "draw ``CATALOG_SIZE``
instances once, up front, and deal them round-robin across the environments". What does
vary per episode is mass, friction and which stable pose the object is resting in.

A catalog entry is a *shape*, not an *identity*: nothing downstream is keyed on the index.
The privileged vector gets the category and the measured half-extents, and that is a
complete description of a convex primitive -- which is why the RMA paper's instance
dictionary is not reproduced here (see doc/design/arm_lift_rma_design.md).

Sizes are bounded by the hardware at both ends: the gripper spans 0.088 m, so the
narrowest cross-section has to close inside that with margin, and the depth camera is
80x60 at roughly 0.5 m, so an object much under 2 cm stops being more than a few pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg

##
# Catalog composition
##

CATEGORIES = ("cuboid", "cylinder", "capsule", "sphere", "cone")
"""Spawner primitive families, in the order the one-hot uses.

A cube is a cuboid whose three half-extents happened to come out equal; it is not its own
category. The line is drawn where Isaac Lab draws it, at the spawner class, so there is no
judgement call about how different two boxes have to be.
"""

NUM_CATEGORIES = len(CATEGORIES)

PER_CATEGORY = 40
"""Instances drawn per category. 40 x 5 = 200 prototypes over (typically) 2048 envs.

High enough that the size distribution is effectively continuous, low enough that scene
creation stays bounded -- every prototype is a separate prim hierarchy, and
``replicate_physics`` is already off.
"""

CATALOG_SIZE = NUM_CATEGORIES * PER_CATEGORY

CATALOG_SEED = 20261014
"""Fixed so the catalog is identical across runs, resumes and evaluation.

The object set is part of the task definition, not part of the training randomness; a
policy and the numbers measured against it have to refer to the same 200 objects.
"""

NOMINAL_DENSITY = 1000.0
"""kg/m^3 before the per-episode multiplier. Mass randomisation scales this at reset."""

##
# Dimension ranges (metres). All are half-extents or radii unless named otherwise.
##

CUBOID_MIN_HALF = (0.010, 0.022)
"""The *narrowest* half-extent, i.e. half the width the gripper has to close on."""
CUBOID_ASPECT = (1.0, 2.2)
"""Ratio of each of the other two half-extents to the narrowest one."""
CUBOID_MAX_HALF = 0.040

CYLINDER_RADIUS = (0.010, 0.022)
CYLINDER_HEIGHT = (0.030, 0.070)

CAPSULE_RADIUS = (0.010, 0.020)
CAPSULE_HEIGHT = (0.020, 0.050)
"""Cylindrical section only; total height is ``height + 2 * radius``."""

SPHERE_RADIUS = (0.014, 0.030)

CONE_RADIUS = (0.016, 0.032)
CONE_HEIGHT = (0.038, 0.070)
"""Chosen so the half-angle ``atan(r/h)`` spans roughly 15-40 deg.

The range was picked from a prediction that did not survive contact with a measurement, and
it is worth writing down which half of it did.

The prediction: a cone presents a wedge to a parallel gripper, a straight squeeze holds only
while ``tan(alpha) < mu``, and over this range the boundary falls at mu = 0.28 to 0.84 --
inside the friction randomisation. Success should therefore track the friction each episode
drew, which nothing can observe. That was the argument for including cones at all, and for
``object_friction`` earning its slot in the privileged vector.

The measurement (model_4999, 1201 cone episodes, placed within 5 cm):

    mu 0.50-0.65   80.5%        mu 0.80-0.95   77.6%
    mu 0.65-0.80   76.6%        mu 0.95-1.10   78.0%

Flat, with the *lowest* friction bin marginally the best -- the opposite of the prediction.
Spheres behave the same way. Whatever is costing cones their 20 points, it is not the
friction draw.

What does correlate is the resting pose: base down, which is the attitude that presents the
wedge, scores 72.8% against 83.3% on its side. So the geometry is implicated and the
coefficient is not. Grip force is the obvious suspect -- the continuous gripper can push
hard enough that mu stops being the binding constraint anywhere in [0.50, 1.10] -- but that
has not been tested, and ``object_friction`` is currently a privileged input with no
measured effect behind it.
"""

##
# Stable resting poses
##


def _quat_about_x(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def _quat_about_y(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

MAX_REST_POSES = 3
"""Cuboids have the most: one per pair of faces."""


@dataclass(frozen=True)
class CatalogEntry:
    """One prototype: how to spawn it, how big it is, and how it can sit on the table."""

    category: str
    spawn: sim_utils.ShapeCfg
    half_extent: tuple[float, float, float]
    """Half-extents along the object's own x, y, z. This is the privileged ``e^dim``."""
    rest_poses: tuple[tuple[tuple[float, float, float, float], float], ...]
    """``(quaternion wxyz, centre height above the table surface)`` per stable pose."""
    volume: float

    @property
    def category_index(self) -> int:
        return CATEGORIES.index(self.category)

    @property
    def nominal_mass(self) -> float:
        return self.volume * NOMINAL_DENSITY


def _shape_kwargs() -> dict:
    """Physics properties shared by every primitive in the catalog."""
    return {
        "rigid_props": RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(density=NOMINAL_DENSITY),
        "collision_props": CollisionPropertiesCfg(),
    }


def _colour(category: str) -> sim_utils.PreviewSurfaceCfg:
    """One colour per family, so a play window is readable at a glance."""
    table = {
        "cuboid": (0.80, 0.25, 0.25),
        "cylinder": (0.25, 0.65, 0.35),
        "capsule": (0.30, 0.45, 0.85),
        "sphere": (0.85, 0.70, 0.20),
        "cone": (0.65, 0.35, 0.75),
    }
    return sim_utils.PreviewSurfaceCfg(diffuse_color=table[category], metallic=0.2)


def _make_cuboid(rng: np.random.Generator) -> CatalogEntry:
    narrow = float(rng.uniform(*CUBOID_MIN_HALF))
    others = sorted(
        min(narrow * float(rng.uniform(*CUBOID_ASPECT)), CUBOID_MAX_HALF) for _ in range(2)
    )
    half = (narrow, others[0], others[1])
    size = tuple(2.0 * h for h in half)
    # One pose per pair of opposite faces. Rotating 90 deg about x brings the local y axis
    # up, about y brings local x up, so the resting height is that axis's half-extent.
    rest = (
        (IDENTITY_QUAT, half[2]),
        (_quat_about_x(math.pi / 2), half[1]),
        (_quat_about_y(math.pi / 2), half[0]),
    )
    return CatalogEntry(
        category="cuboid",
        spawn=sim_utils.CuboidCfg(size=size, visual_material=_colour("cuboid"), **_shape_kwargs()),
        half_extent=half,
        rest_poses=rest,
        volume=size[0] * size[1] * size[2],
    )


def _make_cylinder(rng: np.random.Generator) -> CatalogEntry:
    radius = float(rng.uniform(*CYLINDER_RADIUS))
    height = float(rng.uniform(*CYLINDER_HEIGHT))
    # Upright, or on its side -- where it is in neutral equilibrium about its own axis and
    # will roll away from the fingers if they touch it off-centre.
    rest = ((IDENTITY_QUAT, height / 2), (_quat_about_x(math.pi / 2), radius))
    return CatalogEntry(
        category="cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=radius, height=height, axis="Z", visual_material=_colour("cylinder"), **_shape_kwargs()
        ),
        half_extent=(radius, radius, height / 2),
        rest_poses=rest,
        volume=math.pi * radius**2 * height,
    )


def _make_capsule(rng: np.random.Generator) -> CatalogEntry:
    radius = float(rng.uniform(*CAPSULE_RADIUS))
    height = float(rng.uniform(*CAPSULE_HEIGHT))
    half_z = height / 2 + radius
    rest = ((IDENTITY_QUAT, half_z), (_quat_about_x(math.pi / 2), radius))
    return CatalogEntry(
        category="capsule",
        spawn=sim_utils.CapsuleCfg(
            radius=radius, height=height, axis="Z", visual_material=_colour("capsule"), **_shape_kwargs()
        ),
        half_extent=(radius, radius, half_z),
        rest_poses=rest,
        volume=math.pi * radius**2 * height + 4 / 3 * math.pi * radius**3,
    )


def _make_sphere(rng: np.random.Generator) -> CatalogEntry:
    radius = float(rng.uniform(*SPHERE_RADIUS))
    return CatalogEntry(
        category="sphere",
        spawn=sim_utils.SphereCfg(radius=radius, visual_material=_colour("sphere"), **_shape_kwargs()),
        half_extent=(radius, radius, radius),
        rest_poses=((IDENTITY_QUAT, radius),),
        volume=4 / 3 * math.pi * radius**3,
    )


def _make_cone(rng: np.random.Generator) -> CatalogEntry:
    radius = float(rng.uniform(*CONE_RADIUS))
    height = float(rng.uniform(*CONE_HEIGHT))
    # USD's cone is centred on its own origin: base at -h/2, apex at +h/2.
    #
    # Lying on its side it rests along a slant line, which is horizontal only when the axis
    # is tilted *down* by alpha = atan(r/h) -- apex on the table, fat end up. Solving
    # "apex height == lowest base-rim height" gives that tilt and puts the origin
    # (h/2)*sin(alpha) above the surface. Rotating about y by (90 deg + alpha) maps the
    # local +z axis onto (cos a, 0, -sin a), which is that attitude.
    alpha = math.atan2(radius, height)
    rest = (
        (IDENTITY_QUAT, height / 2),
        (_quat_about_y(math.pi / 2 + alpha), height / 2 * math.sin(alpha)),
    )
    return CatalogEntry(
        category="cone",
        spawn=sim_utils.ConeCfg(
            radius=radius, height=height, axis="Z", visual_material=_colour("cone"), **_shape_kwargs()
        ),
        half_extent=(radius, radius, height / 2),
        rest_poses=rest,
        volume=math.pi * radius**2 * height / 3,
    )


_BUILDERS = {
    "cuboid": _make_cuboid,
    "cylinder": _make_cylinder,
    "capsule": _make_capsule,
    "sphere": _make_sphere,
    "cone": _make_cone,
}


def build_catalog(seed: int = CATALOG_SEED, per_category: int = PER_CATEGORY) -> list[CatalogEntry]:
    """Draw the catalog.

    Interleaved by category rather than grouped, because ``MultiAssetSpawnerCfg`` deals the
    list out round-robin: grouped, a run with few environments would see only cuboids.
    """
    rng = np.random.default_rng(seed)
    per_cat: dict[str, list[CatalogEntry]] = {
        name: [_BUILDERS[name](rng) for _ in range(per_category)] for name in CATEGORIES
    }
    catalog: list[CatalogEntry] = []
    for i in range(per_category):
        for name in CATEGORIES:
            catalog.append(per_cat[name][i])
    return catalog


def catalog_spawn_cfg(catalog: list[CatalogEntry]) -> sim_utils.MultiAssetSpawnerCfg:
    """One prim per environment, dealt deterministically from the catalog.

    ``random_choice`` is False so environment ``i`` always holds ``catalog[i % len]``; the
    privileged observations index the same way and would otherwise describe the wrong
    object. Every property is set on the individual assets, so the wrapper's own fields
    stay None -- a value here would overwrite all of them.
    """
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[entry.spawn for entry in catalog],
        random_choice=False,
    )
