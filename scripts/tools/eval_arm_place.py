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

"""Failure modes and per-condition success rates for the ``Arm-Place-*`` tasks.

    python scripts/tools/eval_arm_place.py --load_run <run> --checkpoint model_1999.pt [--clean_obs]
    python scripts/tools/eval_arm_place.py --task Arm-Place-Cube-Box --load_run <run> \
        --checkpoint model_1999.pt
    python scripts/tools/eval_arm_place.py --task Arm-Place-Sphere-Student --num_envs 256 \
        --load_run <run> --checkpoint model_1999.pt

``--clean_obs`` switches off the teacher's object-position noise. The student's observations
are clean already, so compare a student against a teacher measured with it.

The training log reports one success rate. This sorts every failed episode into the one
place it went wrong, then breaks success down by the things the reset randomised: the
spawn layout, the direction from object to bin, mass and friction. It also reports how high
the object was let go of, and whether the hand was inside the box at that moment.

Failure classes, assigned in this order, from the state at the end of the episode:

  dropped           terminated early -- the object left the table
  never lifted      not once above the lift gate (never picked up)
  held in bin       in the bin, still gripped
  off target        in the bin and let go of, but outside ``place_tolerance`` of the centre
  moving in bin     in the bin, let go, still sliding or rolling (spin about z is allowed)
  held above bin    over the bin and still lifted -- carried there, not lowered
  on the rim        over the bin, not lifted, not in it: resting on a wall
  carrying          lifted, but not over the bin
  out after in      on the table, after having been in the bin -- bounced or knocked out
  on the table      on the table, never in the bin -- put down or dropped short

For the successes and the commonest failure it also prints the continuous state at the
end -- object speed and spin, finger-on-object force, gripper opening -- which is how
"moving in bin" was found to be a sphere spinning in place about the vertical.

The same two counting traps as eval_arm_lift.py apply and are handled the same way: the
final state is sampled at ``max_episode_length - 2`` because ``env.step()`` auto-resets,
and early terminations are counted as failures rather than silently dropped.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Arm-Place-Sphere")
parser.add_argument("--num_envs", type=int, default=500)
parser.add_argument("--episodes", type=int, default=12, help="Episodes per environment.")
parser.add_argument("--arm_init_level", type=int, default=10)
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument(
    "--wall_level",
    type=int,
    default=None,
    help="For tasks with wall levels (Arm-Place-Cube-Box-Levels): the level to measure at."
    " Default: the top one.",
)
parser.add_argument(
    "--clean_obs",
    action="store_true",
    help="Turn off the teacher's object-position noise, to compare it with a student on equal terms.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
# The student task carries a depth camera, which Isaac Lab refuses to spawn without this.
if "student" in args_cli.task.lower():
    args_cli.enable_cameras = True

app = AppLauncher(args_cli).app

import math
import os

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401  (registers the tasks)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from openarm.assets.models.modules.runners import OpenArmDistillationRunner, OpenArmOnPolicyRunner
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.events import object_catalog
from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.rewards import lift_threshold
from openarm.tasks.manager_based.openarm_manipulation.unimanual.place.mdp.rewards import (
    FINGER_OBJECT_SENSORS,
    FINGER_OPEN,
    is_grasped,
    object_in_bin,
    object_is_static,
    object_on_target,
    place_target_w,
    rim_height,
    object_over_bin,
    place_success,
)

RELEASE_STEPS = 3

FAILURES = (
    "dropped",
    "never lifted",
    "held in bin",
    "off target",
    "moving in bin",
    "held above bin",
    "on the rim",
    "carrying",
    "out after in",
    "on the table",
)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.events.reset_robot_joints.params["fixed_level"] = args_cli.arm_init_level
    env_cfg.curriculum.arm_init_level = None
    if args_cli.clean_obs:
        env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg, "wall_levels", None) is not None:
        level = len(env_cfg.wall_levels) - 1 if args_cli.wall_level is None else args_cli.wall_level
        env_cfg.events.reset_bin_walls.params["fixed_level"] = level
        env_cfg.curriculum.wall_level = None
        print(f"[eval] wall level {level}: {env_cfg.wall_levels[level] * 100:.0f} cm", flush=True)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    experiment = agent_cfg.experiment_name or args_cli.task.replace("-Play", "").lower().replace("-", "_")
    # Absolute for the same reason as eval_arm_lift.py: get_checkpoint_path concatenates a
    # relative root with itself.
    root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment))
    checkpoint = get_checkpoint_path(root, args_cli.load_run, args_cli.checkpoint)
    print(f"[eval] checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    runner_cls = (
        OpenArmDistillationRunner if agent_cfg.class_name == "DistillationRunner" else OpenArmOnPolicyRunner
    )
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_module = runner.alg.policy

    u = env.unwrapped
    device = u.device
    n = u.num_envs
    obj, bin_ = u.scene["object"], u.scene["bin"]
    buffers = object_catalog(u)
    margin = u.reward_manager.get_term_cfg("lifting_object").params["margin"]
    snapshot_step = int(u.max_episode_length) - 2
    step_dt = u.step_dt

    def episode_start() -> dict[str, torch.Tensor]:
        """What the reset drew, read right after it -- i.e. for the episode now starting."""
        origins = u.scene.env_origins[:, :2]
        return {
            "obj": obj.data.root_pos_w[:, :2] - origins,
            "bin": bin_.data.root_pos_w[:, :2] - origins,
            "mass": obj.root_physx_view.get_masses().to(device).reshape(n),
            "friction": buffers.friction.clone(),
        }

    start = episode_start()
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=device)
    ever_in_bin = torch.zeros(n, dtype=torch.bool, device=device)
    # Release events. Contact can flicker for a step mid-carry, so a release is a grasp held
    # RELEASE_STEPS steps followed by RELEASE_STEPS steps without one; the state is read at
    # the first of those. The last release of the episode is the one reported.
    geometry = u.cfg.bin_geometry
    grasp_streak = torch.zeros(n, dtype=torch.long, device=device)
    free_streak = torch.zeros(n, dtype=torch.long, device=device)
    armed = torch.zeros(n, dtype=torch.bool, device=device)
    pending_h = torch.zeros(n, device=device)
    pending_in = torch.zeros(n, dtype=torch.bool, device=device)
    release_h = torch.full((n,), float("nan"), device=device)
    release_in = torch.zeros(n, dtype=torch.bool, device=device)
    first_success = torch.full((n,), -1, dtype=torch.long, device=device)
    final: dict[str, torch.Tensor] = {}
    final_detail: dict[str, torch.Tensor] = {}
    robot = u.scene["robot"]
    finger_ids, _ = robot.find_joints("openarm_finger_joint.*")

    def state_detail() -> dict[str, torch.Tensor]:
        """Continuous state at the snapshot, to say *why* a class failed."""
        tcp = u.scene["ee_frame"].data.target_pos_w[:, 0, :]
        contacts = [
            torch.norm(u.scene.sensors[name].data.force_matrix_w, dim=-1).sum(dim=(1, 2))
            for name in FINGER_OBJECT_SENSORS
        ]
        return {
            "lin_vel": torch.norm(obj.data.root_lin_vel_w, dim=1),
            "ang_vel": torch.norm(obj.data.root_ang_vel_w, dim=1),
            "ang_vel_z": obj.data.root_ang_vel_w[:, 2].abs(),
            "finger_force_max": torch.maximum(*contacts),
            "finger_force_min": torch.minimum(*contacts),
            "opening": robot.data.joint_pos[:, finger_ids].mean(dim=1) / FINGER_OPEN,
            "tcp_to_obj": torch.norm(tcp - obj.data.root_pos_w, dim=1),
            # Object height above its resting height in the bin: how far it still had to go down.
            "obj_above_rest": obj.data.root_pos_w[:, 2] - place_target_w(u)[:, 2],
            "obj_xy_off": torch.norm(obj.data.root_pos_w[:, :2] - place_target_w(u)[:, :2], dim=1),
        }

    rows: list[dict] = []
    obs = env.get_observations()
    target = args_cli.episodes * n
    with torch.inference_mode():
        while len(rows) < target:
            obs, _, dones, _ = env.step(policy(obs))
            # The student is recurrent. Training clears its hidden state for the environments
            # that just reset, so evaluation has to as well, or every episode after the first
            # starts out remembering the previous one. A no-op for the feed-forward teacher.
            policy_module.reset(dones)

            lifted = obj.data.root_pos_w[:, 2] > lift_threshold(u, margin)
            in_bin = object_in_bin(u)
            success = place_success(u)
            ever_lifted |= lifted
            ever_in_bin |= in_bin
            new_success = success & (first_success < 0)
            first_success[new_success] = u.episode_length_buf[new_success]

            grasped = is_grasped(u)
            grasp_streak = torch.where(grasped, grasp_streak + 1, torch.zeros_like(grasp_streak))
            armed |= grasp_streak >= RELEASE_STEPS
            letting_go = armed & ~grasped & (free_streak == 0)
            if letting_go.any():
                # Height of the sphere's centre above where it rests on the bin floor, and
                # whether the TCP was inside the box, below the rim, at that moment.
                rest_w = place_target_w(u)
                tcp = u.scene["ee_frame"].data.target_pos_w[:, 0, :]
                tcp_off = tcp - bin_.data.root_pos_w
                inside = (tcp_off[:, :2].abs() <= geometry.inner_half).all(dim=1) & (
                    tcp_off[:, 2] <= rim_height(u)
                )
                pending_h = torch.where(letting_go, obj.data.root_pos_w[:, 2] - rest_w[:, 2], pending_h)
                pending_in = torch.where(letting_go, inside, pending_in)
            free_streak = torch.where(armed & ~grasped, free_streak + 1, torch.zeros_like(free_streak))
            committed = free_streak == RELEASE_STEPS
            release_h = torch.where(committed, pending_h, release_h)
            release_in = torch.where(committed, pending_in, release_in)
            armed &= ~committed

            at_snapshot = u.episode_length_buf == snapshot_step
            if at_snapshot.any():
                now = {
                    "success": success,
                    "in_bin": in_bin,
                    "on_target": object_on_target(u),
                    "grasped": is_grasped(u),
                    "static": object_is_static(u),
                    "over_bin": object_over_bin(u),
                    "lifted": lifted,
                    "seen": torch.ones(n, dtype=torch.bool, device=device),
                }
                detail = state_detail()
                for key, value in detail.items():
                    final_detail[key] = torch.where(at_snapshot, value, final_detail.get(key, torch.zeros_like(value)))
                for key, value in now.items():
                    final[key] = torch.where(at_snapshot, value, final.get(key, torch.zeros_like(value)))

            done = dones.bool()
            if done.any():
                seen = final.get("seen", torch.zeros(n, dtype=torch.bool, device=device))
                for i in done.nonzero(as_tuple=False).flatten().tolist():
                    f = {k: bool(v[i]) for k, v in final.items()} if bool(seen[i]) else {}
                    rows.append(
                        {
                            "obj": start["obj"][i].tolist(),
                            "bin": start["bin"][i].tolist(),
                            "mass": float(start["mass"][i]),
                            "friction": float(start["friction"][i]),
                            "outcome": classify(f, bool(ever_lifted[i]), bool(ever_in_bin[i])),
                            "pushed": bool(f.get("success")) and not bool(ever_lifted[i]),
                            "t_success": int(first_success[i]) * step_dt if first_success[i] >= 0 else None,
                            "release_h": None if torch.isnan(release_h[i]) else float(release_h[i]),
                            "release_in": bool(release_in[i]),
                            "detail": {k: float(v[i]) for k, v in final_detail.items()} if bool(seen[i]) else {},
                        }
                    )
                fresh = episode_start()
                for key in start:
                    start[key][done] = fresh[key][done]
                ever_lifted[done] = False
                ever_in_bin[done] = False
                first_success[done] = -1
                grasp_streak[done] = 0
                free_streak[done] = 0
                armed[done] = False
                release_h[done] = float("nan")
                release_in[done] = False
                for key in final:
                    final[key][done] = False

    rows = rows[:target]
    report(rows)
    env.close()


def classify(f: dict, ever_lifted: bool, ever_in_bin: bool) -> str:
    if not f:
        return "dropped"
    if f["success"]:
        return "success"
    if not ever_lifted:
        return "never lifted"
    if f["in_bin"]:
        if f["grasped"]:
            return "held in bin"
        return "off target" if not f["on_target"] else "moving in bin"
    if f["over_bin"]:
        return "held above bin" if f["lifted"] else "on the rim"
    if f["lifted"]:
        return "carrying"
    return "out after in" if ever_in_bin else "on the table"


def report(rows: list[dict]) -> None:
    total = len(rows)
    ok = sum(r["outcome"] == "success" for r in rows)
    print(f"\n[eval] {total} episodes, arm_init_level {args_cli.arm_init_level}"
          + (", clean observations" if args_cli.clean_obs else ""))
    print(f"[eval] success {ok / total * 100:.2f}%  ({total - ok} failures)")

    times = sorted(r["t_success"] for r in rows if r["t_success"] is not None)
    if times:
        q = lambda p: times[min(len(times) - 1, int(p * len(times)))]  # noqa: E731
        print(f"[eval] time to first success: median {q(0.5):.2f} s, p90 {q(0.9):.2f} s, p99 {q(0.99):.2f} s")
    pushed = sum(r["pushed"] for r in rows)
    print(f"[eval] successes without ever lifting (pushed in): {pushed}")

    print("\n" + "=" * 60)
    print("RELEASE: object centre above its resting height, at the last release")
    print("=" * 60)
    edges = (-1.0, 0.01, 0.03, 0.06, 0.10, 1.0)
    def band(lo: float, hi: float) -> str:
        if lo < 0:
            return f"<{hi * 100:.0f} cm"
        return f">{lo * 100:.0f} cm" if hi >= 1 else f"{lo * 100:.0f}-{hi * 100:.0f} cm"

    print(f"{'':<16}{'n':>6}{'hand in box':>13}" + "".join(
        f"{band(lo, hi):>11}" for lo, hi in zip(edges[:-1], edges[1:])))
    for label, group in (
        ("success", [r for r in rows if r["outcome"] == "success"]),
        ("failure", [r for r in rows if r["outcome"] != "success"]),
    ):
        released = [r for r in group if r["release_h"] is not None]
        if not released:
            continue
        m = len(released)
        cells = "".join(
            f"{sum(lo <= r['release_h'] < hi for r in released) / m * 100:>10.1f}%"
            for lo, hi in zip(edges[:-1], edges[1:])
        )
        print(f"{label:<16}{m:>6}{sum(r['release_in'] for r in released) / m * 100:>12.1f}%{cells}")

    print("\n" + "=" * 60)
    print("FAILURE MODES (share of all episodes / of failures)")
    print("=" * 60)
    for name in FAILURES:
        k = sum(r["outcome"] == name for r in rows)
        if k:
            print(f"{name:<20}{k:>7}{k / total * 100:>9.2f}%{k / max(1, total - ok) * 100:>9.1f}%")

    print("\n" + "=" * 60)
    print("STATE AT THE END: median [p10, p90]")
    print("=" * 60)
    keys = ("lin_vel", "ang_vel", "ang_vel_z", "finger_force_max", "finger_force_min", "opening", "tcp_to_obj",
            "obj_above_rest", "obj_xy_off")
    print(f"{'':<16}" + "".join(f"{k:>20}" for k in keys))
    for name in ("success", "moving in bin", "held above bin", "held in bin", "off target"):
        group = [r["detail"] for r in rows if r["outcome"] == name and r["detail"]]
        if not group:
            continue
        cells = []
        for k in keys:
            v = sorted(d[k] for d in group)
            pick = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
            cells.append(f"{pick(0.5):.3f} [{pick(0.1):.2f},{pick(0.9):.2f}]")
        print(f"{name:<16}" + "".join(f"{c:>20}" for c in cells))
    moving = [r["detail"] for r in rows if r["outcome"] == "moving in bin" and r["detail"]]
    if moving:
        m = len(moving)
        print(f"moving in bin, {m} episodes: lin_vel > 0.01 {sum(d['lin_vel'] > 0.01 for d in moving) / m * 100:.1f}%, "
              f"ang_vel > 0.5 {sum(d['ang_vel'] > 0.5 for d in moving) / m * 100:.1f}%, "
              f"a finger touching (>0.5 N) {sum(d['finger_force_max'] > 0.5 for d in moving) / m * 100:.1f}%")

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def angle(r):
        """Direction from sphere to bin, relative to the robot: 0 = straight away from it."""
        dx, dy = r["bin"][0] - r["obj"][0], r["bin"][1] - r["obj"][1]
        a = math.degrees(math.atan2(dy, dx))
        if -45 <= a < 45:
            return "away from base"
        if 45 <= a < 135:
            return "to the left (+y)"
        if -135 <= a < -45:
            return "to the right (-y)"
        return "toward base"

    breakdowns = {
        "SPHERE-TO-BIN DISTANCE": (lambda r: dist(r["obj"], r["bin"]), (0.08, 0.11, 0.14, 0.17, 0.21), "{:.2f}-{:.2f} m"),
        "SPHERE DISTANCE FROM BASE": (lambda r: math.hypot(*r["obj"]), (0.28, 0.34, 0.38, 0.42, 0.46), "{:.2f}-{:.2f} m"),
        "BIN DISTANCE FROM BASE": (lambda r: math.hypot(*r["bin"]), (0.28, 0.34, 0.38, 0.42, 0.46), "{:.2f}-{:.2f} m"),
        "SPHERE Y": (lambda r: r["obj"][1], (-0.19, -0.09, 0.0, 0.09, 0.19), "{:+.2f}..{:+.2f} m"),
        "BIN Y": (lambda r: r["bin"][1], (-0.19, -0.09, 0.0, 0.09, 0.19), "{:+.2f}..{:+.2f} m"),
        "MASS": (lambda r: r["mass"] * 1000, (15, 50, 85, 120, 170), "{:.0f}-{:.0f} g"),
        "FRICTION": (lambda r: r["friction"], (0.50, 0.65, 0.80, 0.95, 1.10), "{:.2f}-{:.2f}"),
    }

    def line(label, group):
        m = len(group)
        if not m:
            return
        s = sum(r["outcome"] == "success" for r in group)
        top = {}
        for r in group:
            if r["outcome"] != "success":
                top[r["outcome"]] = top.get(r["outcome"], 0) + 1
        worst = ", ".join(f"{k} {v / m * 100:.1f}%" for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:2])
        print(f"{label:<22}{m:>6}{s / m * 100:>9.2f}%   {worst}")

    print("\n" + "=" * 60)
    print("BY DIRECTION, SPHERE TO BIN")
    print("=" * 60)
    for label in ("away from base", "toward base", "to the left (+y)", "to the right (-y)"):
        line(label, [r for r in rows if angle(r) == label])

    for title, (key, edges, fmt) in breakdowns.items():
        print("\n" + "=" * 60)
        print(f"BY {title}")
        print("=" * 60)
        for lo, hi in zip(edges[:-1], edges[1:]):
            line(fmt.format(lo, hi), [r for r in rows if lo <= key(r) < hi])
    print("=" * 60)


main()
app.close()
