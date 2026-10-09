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

"""Scratch space for one-off ablations, plus the record of the ones already run.

Nothing is registered right now. Drop a ``tryN.py`` in here defining ``TryNEnvCfg`` and
it is picked up automatically as ``Arm-Lift-Cube-TryN``; the files themselves are
gitignored, because a try is worth keeping only until its question is answered and what
it measured belongs in SUMMARY.md.

Two rules, both learned the hard way and both recorded in SUMMARY.md:

* Pass ``--seed`` and run more than one. Identical configurations here have landed at
  ``lifting_object`` 11.27 and 0.127 on different seeds, and single runs produced two
  diagnoses this session that later measurements overturned.
* State every knob a try depends on explicitly rather than inheriting it. unitree_rl_lab
  lost a run to a sandbox try that silently picked up later edits to the config it
  built on.

    python -u scripts/rsl_rl/train.py --task Arm-Lift-Cube-Try0 \
        --headless --seed 42 --max_iterations 1000

Logs land in ``logs/rsl_rl/arm_lift_cube_try<N>/``.
"""

import glob
import os
import re

import gymnasium as gym

from .. import agents

_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg"

for _path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "try*.py"))):
    _match = re.fullmatch(r"try(\d+)", os.path.splitext(os.path.basename(_path))[0])
    if _match is None:
        continue
    _n = _match.group(1)
    gym.register(
        id=f"Arm-Lift-Cube-Try{_n}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.try{_n}:Try{_n}EnvCfg",
            "rsl_rl_cfg_entry_point": _AGENT,
        },
        disable_env_checker=True,
    )
