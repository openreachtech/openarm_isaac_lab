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

"""Ablation sandbox: which change stopped the teacher learning?

Run `2026-10-05_23-55-28` (2048 envs, 1000 iterations) never lifted the cube --
`lifting_object` sat at its random-policy value of 0.12 from iteration 0 to 999 while
`reaching_object` rose to 0.64 and then collapsed. Four things had changed at once
since the last good teacher, plus the batch size. These tries turn them on one at a
time.

| try | obs noise | cube yaw | cube scale | arm init | curriculum | num_envs |
|-----|-----------|----------|------------|----------|------------|----------|
|  0  |     -     |    -     |     -      |    -     |   10000    |   2048   |
|  1  |   0.02    |    -     |     -      |    -     |   10000    |   2048   |
|  2  |     -     |   +-pi   |     -      |    -     |   10000    |   2048   |
|  3  |     -     |    -     | 0.72-0.88  |    -     |   10000    |   2048   |
|  4  |     -     |    -     |     -      | 0.1 rad  |   10000    |   2048   |
|  5  |   0.02    |   +-pi   | 0.72-0.88  | 0.1 rad  |   10000    |   2048   |
|  6  |   0.02    |   +-pi   | 0.72-0.88  | 0.1 rad  |   30000    |   2048   |
|  7  |     -     |    -     |     -      |    -     |   10000    |   4096   |
|  8  |   0.02    |   +-pi   | 0.72-0.88  |    -     |   10000    |   2048   |
|  9  |   0.02    |   +-pi   | 0.72-0.88  |    -     |   30000    |   2048   |

Read try0 and try7 first: they carry no randomisation at all, so if they fail the
cause is the batch size and its interaction with the curriculum, not the
randomisation, and tries 1-4 cannot be interpreted on their own.

Always pass ``--seed 42``. unitree_rl_lab's sandbox lineage drew wrong conclusions
from unseeded single runs -- a fresh retrain of a known-good config landed markedly
worse purely through seed variance (see its sandbox/SUMMARY.md). A shared seed does
not remove PPO variance, but it removes seed choice as a difference between tries.

    python -u scripts/rsl_rl/train.py --task Arm-Lift-Cube-Try0 \
        --headless --seed 42 --max_iterations 1000

Logs land in ``logs/rsl_rl/arm_lift_cube_try<N>/``.

The tryN.py files themselves are gitignored: they are scratch, and what they measured
belongs in SUMMARY.md. This module and ablation.py are committed so the scaffolding
survives. Write the result into SUMMARY.md before deleting a try.
"""

import glob
import os
import re

import gymnasium as gym

from .. import agents

_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg"

# The tryN.py files are gitignored (see .gitignore), so register whichever are present
# rather than a fixed range -- a checkout without any registers nothing and still imports.
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
