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

"""The bin for ``Arm-Place-Sphere``: a floor plate and four low walls, as one rigid body.

Dimensions are ManiSkill's ``PlaceSphere-v1`` (``_build_bin``): a 4 cm square pocket, walls
5 mm thick and 1 cm tall standing on a 5 mm floor, so a 2 cm sphere fits it exactly.

Isaac Lab's shape spawners make one shape per prim, and a rigid body with five colliders is
what is needed here -- five separate bodies would each have to be placed at reset and could
drift apart. So the USD is written out with pxr and spawned through ``UsdFileCfg``.
"""

from __future__ import annotations

import os
import tempfile

BIN_INNER_HALF = 0.02
"""Half the pocket's inner side. Equal to the sphere radius, as in ManiSkill.

The first thing to widen (to 0.025, 5 mm of play) if placement precision will not come up.
"""

BIN_WALL_THICKNESS = 0.005
BIN_WALL_HEIGHT = 0.01
BIN_FLOOR_HALF_THICKNESS = 0.0025
"""The body's origin is the centre of the floor plate, so its top is this far above it."""

BIN_OUTER_HALF = BIN_INNER_HALF + BIN_WALL_THICKNESS

BIN_COLOUR = (0.55, 0.45, 0.35)


def _boxes() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """``(name, centre, half-size)`` of each plate, in the body frame."""
    wall_offset = BIN_INNER_HALF + BIN_WALL_THICKNESS / 2
    wall_z = BIN_FLOOR_HALF_THICKNESS + BIN_WALL_HEIGHT / 2
    wall_half = BIN_WALL_HEIGHT / 2
    thin = BIN_WALL_THICKNESS / 2
    return [
        ("floor", (0.0, 0.0, 0.0), (BIN_OUTER_HALF, BIN_OUTER_HALF, BIN_FLOOR_HALF_THICKNESS)),
        ("wall_px", (wall_offset, 0.0, wall_z), (thin, BIN_OUTER_HALF, wall_half)),
        ("wall_nx", (-wall_offset, 0.0, wall_z), (thin, BIN_OUTER_HALF, wall_half)),
        ("wall_py", (0.0, wall_offset, wall_z), (BIN_OUTER_HALF, thin, wall_half)),
        ("wall_ny", (0.0, -wall_offset, wall_z), (BIN_OUTER_HALF, thin, wall_half)),
    ]


def bin_usd_path() -> str:
    """Write the bin's USD if it is not there yet, and return its path.

    Cached under the system temp directory with the dimensions in the name, so a change to
    any of them produces a new file rather than silently reusing the old one. Written to a
    private name and renamed into place, so two processes starting together cannot read a
    half-written file.

    Needs pxr, so call it after the simulation app has started -- from an env cfg's
    ``__post_init__``, not at import time.
    """
    tag = "_".join(
        f"{v * 1000:g}"
        for v in (BIN_INNER_HALF, BIN_WALL_THICKNESS, BIN_WALL_HEIGHT, BIN_FLOOR_HALF_THICKNESS)
    )
    directory = os.path.join(tempfile.gettempdir(), "openarm")
    path = os.path.join(directory, f"place_bin_{tag}.usda")
    if os.path.exists(path):
        return path

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    os.makedirs(directory, exist_ok=True)
    scratch = f"{path}.{os.getpid()}.usda"
    stage = Usd.Stage.CreateNew(scratch)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Bin")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    # Kinematic, so the mass never enters the dynamics; set only to keep PhysX from
    # computing one from five colliders and warning about it.
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1)

    for name, centre, half in _boxes():
        cube = UsdGeom.Cube.Define(stage, f"/Bin/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*centre))
        xform.SetScale(Gf.Vec3f(*(2.0 * h for h in half)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*BIN_COLOUR)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage.GetRootLayer().Save()
    os.replace(scratch, path)
    return path
