#!/usr/bin/env python3
"""Run training then aggregate TensorBoard logs with quiet subprocess output.

The full train.py output is saved to ``<run_dir>/train.log``; only the aggregated summary is printed.
Unknown arguments are forwarded to train.py as-is (e.g. ``--resume --load_run <run> --checkpoint <ckpt>``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train and aggregate logs in one command.")
    parser.add_argument("--task", required=True, help="Task ID passed to train.py")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Training iterations for train.py (default: the task's agent config).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments (default: the task's env config).",
    )
    parser.add_argument(
        "--aggregate-interval",
        type=int,
        default=100,
        help="Iteration interval for aggregate_tensorboard_logs.py (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed passed to train.py. rsl_rl defaults to 42 if unset, so repeated runs of the "
        "same config are otherwise byte-for-byte identical, not independent trials.",
    )
    return parser.parse_known_args()


def tail(path: Path, num_chars: int = 4000) -> str:
    return path.read_text(errors="replace")[-num_chars:]


def run_dir_from_train_log(train_log: Path) -> Path:
    """Locate the run directory from the log lines printed by train.py."""
    text = train_log.read_text(errors="replace")
    root_match = re.search(r"\[INFO\] Logging experiment in directory: (.+)", text)
    name_match = re.search(r"Exact experiment name requested from command line: (\S+)", text)
    if not root_match or not name_match:
        raise RuntimeError(f"Could not find the run directory in train.py output: {train_log}")
    log_root = Path(root_match.group(1).strip())
    # run_name may be appended to the timestamp, so match by prefix.
    runs = sorted(log_root.glob(f"{name_match.group(1)}*"))
    if not runs:
        raise FileNotFoundError(f"Run directory not found under: {log_root}")
    return runs[-1]


def latest_summary_file(run_dir: Path) -> Path:
    summaries = sorted(run_dir.glob("events_log_summary_*.md"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"No aggregate markdown found in: {run_dir}")
    return summaries[-1]


def main() -> None:
    args, train_extra_args = parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train.py"
    aggregate_script = script_dir / "aggregate_tensorboard_logs.py"

    train_cmd = [sys.executable, str(train_script), "--task", args.task, "--headless"]
    if args.num_envs is not None:
        train_cmd.extend(["--num_envs", str(args.num_envs)])
    if args.max_iterations is not None:
        train_cmd.extend(["--max_iterations", str(args.max_iterations)])
    if args.seed is not None:
        train_cmd.extend(["--seed", str(args.seed)])
    train_cmd.extend(train_extra_args)

    # The run directory is unknown until train.py starts, so log to a temp file and move it afterwards.
    log_base = Path("logs", "rsl_rl").resolve()
    log_base.mkdir(parents=True, exist_ok=True)
    train_log = log_base / f".train_and_aggregate_{os.getpid()}.log"
    with train_log.open("w") as log_file:
        result = subprocess.run(train_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    try:
        run_dir = run_dir_from_train_log(train_log)
        train_log = train_log.rename(run_dir / "train.log")
    except (RuntimeError, FileNotFoundError):
        run_dir = None

    if result.returncode != 0 or run_dir is None:
        raise RuntimeError(
            f"train.py failed with exit code {result.returncode}.\n"
            f"Full log: {train_log}\n"
            f"Last output:\n{tail(train_log)}"
        )

    aggregate_cmd = [
        sys.executable,
        str(aggregate_script),
        "--logdir",
        str(run_dir),
        "--interval",
        str(args.aggregate_interval),
        "--overwrite",
    ]
    result = subprocess.run(aggregate_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"aggregate_tensorboard_logs.py failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )

    summary_path = latest_summary_file(run_dir)
    print(f"[Run Dir] {run_dir}")
    print(f"[Train Log] {train_log}")
    print(f"[Summary File] {summary_path}")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
