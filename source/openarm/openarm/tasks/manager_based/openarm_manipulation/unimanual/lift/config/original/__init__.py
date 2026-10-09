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

"""The lift task as it stood at the first commit, kept as a reference baseline.

``Arm-Lift-Cube`` started life as this task and has since diverged a long way: the arm
base was raised for the quadruped mount, the cube gained yaw and size randomisation, the
teacher's object-position observation gained noise, and the spawn box was narrowed after
measuring that grasping fails past about 0.45 m from the base. Those changes were all
deliberate, but they make it impossible to answer "did this get better or worse than
where we started" from the live task alone.

Keeping it paid for itself immediately. Replaying this task beside the live one is what
showed the trembling came from the base height rather than from the action scaling, and
sweeping the height over this config is what found the cliff between 0.10 and 0.15 --
see config/sandbox/SUMMARY.md.

The files here are verbatim copies from commit 2d1794e with only their relative imports
adjusted. They deliberately do **not** inherit from ``lift/lift_env_cfg.py`` -- sharing
that base would mean every later edit to the live task silently rewrote this baseline
too, which is exactly the trap unitree_rl_lab records in its sandbox/SUMMARY.md. Leave
them frozen; if the baseline needs to change, that is a new task, not an edit here.

Logs land in ``logs/rsl_rl/isaac_lift_cube_openarm_v0/``.
"""

import gymnasium as gym

from .. import agents

gym.register(
    id="Isaac-Lift-Cube-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
