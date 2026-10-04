---
name: train-and-check
description: Run RSL-RL training for an OpenArm task and print aggregated TensorBoard results. Use when the user asks to train a task and check results, or says "train and aggregate", "train and check", or mentions train_and_aggregate.py.
---

# Train and Check Results

Runs `scripts/rsl_rl/train_and_aggregate.py` and prints the aggregated metric summary.
Console output from training is not printed; it is saved to `<run_dir>/train.log`. Only the final summary is printed.

## Arguments

Pass flags directly to `train_and_aggregate.py`:

| Flag | Required | Default | Description |
|---|---|---|---|
| `--task` | Yes | — | Gym task ID (e.g. `Isaac-Lift-Cube-OpenArm-v0`) |
| `--max_iterations` | No | agent cfg | Number of training iterations |
| `--num_envs` | No | env cfg | Number of parallel environments |
| `--seed` | No | 42 (rsl_rl) | Seed; set a different value for independent trials |
| `--aggregate-interval` | No | 100 | Log aggregation interval |

Any other flags (e.g. `--resume --load_run <run> --checkpoint <ckpt>`) are forwarded to `train.py`.
Fixed internally: `--headless`.

Available tasks: `Isaac-Reach-OpenArm-v0`, `Isaac-Lift-Cube-OpenArm-v0`, `Isaac-Open-Drawer-OpenArm-v0`, `Isaac-Reach-OpenArm-Bi-v0`.
Logs and checkpoints go to `logs/rsl_rl/<task_name>/<timestamp>/`, where `<task_name>` is the task ID lowercased with
`-` replaced by `_` (e.g. `logs/rsl_rl/isaac_lift_cube_openarm_v0/`). Play tasks (`...-Play-v0`) resolve to the same directory.

## Steps

1. Identify `--task` from the user's message.
2. If the user wants to continue a previous run, add `--resume` (plus `--load_run` / `--checkpoint` if specified).
3. Pass `--max_iterations` / `--num_envs` / `--seed` only if the user specified them.
4. Run from the repository root with the Isaac Lab venv's Python, in the background (training takes a long time):

```bash
/home/tak/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train_and_aggregate.py --task <TASK> [--max_iterations <N>]
```

5. Print the command output to the user as-is.
6. On failure, the error includes the tail of the training output and the path to the full log; read the log to diagnose.

If `--task` is missing, ask the user before running.

## Example

```bash
/home/tak/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/train_and_aggregate.py \
  --task Isaac-Lift-Cube-OpenArm-v0 \
  --max_iterations 1500
```
