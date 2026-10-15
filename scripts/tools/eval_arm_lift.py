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

"""Per-shape success rates for ``Arm-Lift``.

Four breakdowns of the same episodes: by category, by which stable pose the object was
resting in, by the narrowest dimension the fingers had to close on, and -- for the two
categories that do worst -- by the friction that episode drew. The last one exists because
the catalog's cone sizes were chosen on a prediction about friction, and a table is the only
way to find out whether the prediction held. (It did not; see object_catalog.py.)

Three things this measures that the training log cannot:

* *lifted* -- did the object ever clear its own resting height by the reward's margin.
  Episode-mean reward conflates "lifted briefly" with "lifted and held".
* *placed* -- was it within 5 cm of the goal at the end. The reward's tanh kernel gives
  partial credit; this does not.
* *lost after grasp* -- lifted at some point, but not held to the end. On the cube task
  this turned out to be the whole story: every shape was picked up 100% of the time and
  the differences between them were entirely in what happened afterwards.

Two counting traps, both of which inflated an earlier version of this measurement:

* ``env.step()`` auto-resets, so anything read after it belongs to the *next* episode.
  The final state is sampled at ``max_episode_length - 2``.
* episodes that terminate early never reach that step. Silently dropping them removed the
  worst episodes from the statistics -- cylinder success read 88.3% when the true figure
  over all 800 episodes was lower. They are counted as failures here, explicitly.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Arm-Lift")
parser.add_argument("--num_envs", type=int, default=500)
parser.add_argument("--episodes", type=int, default=12, help="Episodes per environment.")
parser.add_argument("--arm_init_level", type=int, default=10)
parser.add_argument("--place_tolerance", type=float, default=0.05)
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
# The student task carries a depth camera, which Isaac Lab refuses to spawn without this.
if "student" in args_cli.task.lower():
    args_cli.enable_cameras = True

app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401  (registers the tasks)
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from openarm.assets.models.modules.runners import OpenArmDistillationRunner, OpenArmOnPolicyRunner
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.events import object_catalog
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.rewards import lift_threshold
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.object_catalog import CATEGORIES


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # Pin the arm-init curriculum: outside training it restarts at level 0 and would need
    # hundreds of resets to climb, so an unpinned run measures the easiest setting rather
    # than the one the policy was trained to.
    env_cfg.events.reset_robot_joints.params["fixed_level"] = args_cli.arm_init_level
    env_cfg.curriculum.arm_init_level = None

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    # The registry leaves experiment_name empty and train.py fills it from the task name;
    # resolving it here too keeps this script from treating logs/rsl_rl itself as a run.
    experiment = agent_cfg.experiment_name or args_cli.task.replace("-Play", "").lower().replace("-", "_")

    # Absolute, because get_checkpoint_path joins log_path with a DirEntry whose
    # __fspath__ is already the full path: with a relative root the two concatenate into
    # 'logs/rsl_rl/arm_lift/logs/rsl_rl/arm_lift/<run>'. An absolute root makes the second
    # path win, which is the intended behaviour.
    root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment))
    checkpoint = get_checkpoint_path(root, args_cli.load_run, args_cli.checkpoint)
    print(f"[eval] checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    runner_cls = (
        OpenArmDistillationRunner
        if agent_cfg.class_name == "DistillationRunner"
        else OpenArmOnPolicyRunner
    )
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    device = u.device
    n = u.num_envs
    obj = u.scene["object"]
    buffers = object_catalog(u)
    category = buffers.category_onehot.argmax(dim=-1)
    # The narrowest object-frame dimension: the width the fingers have to close on, and the
    # quantity the cube task found the loss-after-grasp rate tracked monotonically.
    narrow = buffers.half_extent.min(dim=-1).values * 2.0
    margin = u.reward_manager.get_term_cfg("lifting_object").params["margin"]
    snapshot_step = int(u.max_episode_length) - 2

    # Per-episode state. friction and the resting attitude are resampled at reset, and
    # env.step() auto-resets, so both are snapshotted *before* each step: at the moment a
    # done is observed they still describe the episode that just ended.
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=device)
    final_dist = torch.full((n,), float("inf"), device=device)
    final_up = torch.zeros(n, dtype=torch.bool, device=device)

    REST_LABELS = {
        "cuboid": ("face z", "face y", "face x"),
        "cylinder": ("upright", "on side"),
        "capsule": ("upright", "on side"),
        "sphere": ("resting",),
        "cone": ("base down", "on side"),
    }
    FRICTION_BINS = (0.50, 0.65, 0.80, 0.95, 1.10)
    WIDTH_BINS = (0.0, 0.025, 0.035, 0.045, 1.0)

    rows: list[dict] = []

    obs = env.get_observations()
    target = args_cli.episodes * n
    steps = 0
    with torch.inference_mode():
        while len(rows) < target:
            friction_now = buffers.friction.clone()
            slot_now = buffers.rest_pose_slot.clone()

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            steps += 1

            threshold = lift_threshold(u, margin)
            above = obj.data.root_pos_w[:, 2] > threshold
            ever_lifted |= above

            at_snapshot = u.episode_length_buf == snapshot_step
            if at_snapshot.any():
                robot = u.scene["robot"]
                goal_b = u.command_manager.get_command("object_pose")[:, :3]
                goal_w, _ = combine_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, goal_b
                )
                dist = torch.norm(goal_w - obj.data.root_pos_w, dim=-1)
                final_dist = torch.where(at_snapshot, dist, final_dist)
                final_up = torch.where(at_snapshot, above, final_up)

            done = dones.bool()
            if done.any():
                idx = done.nonzero(as_tuple=False).flatten()
                placed = ever_lifted & (final_dist < args_cli.place_tolerance)
                for i in idx.tolist():
                    rows.append(
                        {
                            "cat": int(category[i]),
                            "slot": int(slot_now[i]),
                            "friction": float(friction_now[i]),
                            "narrow": float(narrow[i]),
                            "lifted": bool(ever_lifted[i]),
                            "placed": bool(placed[i]),
                            # "held" means it was still above its own lift gate at the end.
                            # Separating this from "placed" is what distinguishes dropping
                            # the object from carrying it to the wrong spot; the first
                            # version of this script reported only their sum.
                            "held": bool(final_up[i]),
                        }
                    )
                ever_lifted[done] = False
                final_dist[done] = float("inf")
                final_up[done] = False

    rows = rows[:target]

    def block(title: str, groups: list[tuple[str, list[dict]]]) -> None:
        print("\n" + "=" * 82)
        print(title)
        print("=" * 82)
        hdr = (f"{'':<22}{'n':>7}{'lifted':>9}{'placed':>9}{'dropped':>10}"
               f"{'misplaced':>12}")
        print(hdr)
        print("-" * 82)
        for label, group in groups:
            if not group:
                continue
            m = len(group)
            lift = sum(r["lifted"] for r in group)
            place = sum(r["placed"] for r in group)
            # Lifted at some point but not holding it at the end: it came back down.
            drop = sum(r["lifted"] and not r["held"] for r in group)
            # Still up, but not close enough.
            mis = sum(r["lifted"] and r["held"] and not r["placed"] for r in group)
            print(f"{label:<22}{m:>7}{lift / m * 100:>8.1f}%{place / m * 100:>8.1f}%"
                  f"{drop / m * 100:>9.1f}%{mis / m * 100:>11.1f}%")

    print(f"\n[eval] {len(rows)} episodes over {steps} steps, "
          f"arm_init_level {args_cli.arm_init_level}, tolerance {args_cli.place_tolerance * 100:.0f} cm")

    block("BY CATEGORY", [(c, [r for r in rows if r["cat"] == i]) for i, c in enumerate(CATEGORIES)]
          + [("ALL", rows)])

    block("BY CATEGORY AND RESTING POSE", [
        (f"{c} / {lab}", [r for r in rows if r["cat"] == i and r["slot"] == s_i])
        for i, c in enumerate(CATEGORIES)
        for s_i, lab in enumerate(REST_LABELS[c])
    ])

    block("BY NARROWEST WIDTH (the dimension the fingers close on)", [
        (f"{lo * 100:.1f}-{min(hi, 0.09) * 100:.1f} cm",
         [r for r in rows if lo <= r["narrow"] < hi])
        for lo, hi in zip(WIDTH_BINS[:-1], WIDTH_BINS[1:])
    ])

    block("CONE AND SPHERE BY EPISODE FRICTION", [
        (f"{CATEGORIES[i]} mu {lo:.2f}-{hi:.2f}",
         [r for r in rows if r["cat"] == i and lo <= r["friction"] < hi])
        for i in (CATEGORIES.index("cone"), CATEGORIES.index("sphere"))
        for lo, hi in zip(FRICTION_BINS[:-1], FRICTION_BINS[1:])
    ])
    print("=" * 82)

    env.close()


main()
app.close()
