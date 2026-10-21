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

import gymnasium as gym

from ...lift.config import agents

##
# Arm-Place-Sphere: the Arm-Lift RMA teacher, putting a sphere in a bin.
##

gym.register(
    id="Arm-Place-Sphere",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_place_sphere_env_cfg:ArmPlaceSphereEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.arm_place_sphere_env_cfg:ArmPlaceSphereEnvCfg_PLAY",
        # Same observation widths and privileged layout as Arm-Lift, so the same runner.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_rma_cfg:ArmLiftRmaPPORunnerCfg",
    },
    disable_env_checker=True,
)
