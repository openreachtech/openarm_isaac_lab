# Sandbox summaries

## Which change stopped the teacher learning? (2026-10-06)

### Why

Four things were added to the lift task at once, aimed at a more robust teacher for
depth-student distillation:

1. `object_position` observation noise, sigma = 0.02 m, per-step plus a per-episode
   bias (DextrAH-G section 3.2 / appendix E.4)
2. cube yaw randomised over +-pi
3. cube scale randomised 0.72-0.88 (edge 3.72-4.68 cm, +-11%)
4. initial arm joint offset +-0.1 rad on the 7 arm joints

`num_envs` also dropped 4096 -> 2048, because (3) forces `replicate_physics = False`
and 4096 then needs ~46 GB of host RAM.

The first training run with all of it on (`2026-10-05_23-55-28`, 1000 iterations) never
lifted the cube: `lifting_object` sat at its random-policy value of 0.12 from iteration
0 to 999.

### Setup

Eight tries, 1000 iterations each, `--seed 42` throughout, `tryN.py` each stating all
six knobs explicitly rather than inheriting them.

### Results

| try | obs noise | yaw | scale | arm init | curric. | envs | lift @399 | final lift | final noise std | verdict |
|-----|-----------|-----|-------|----------|---------|------|-----------|------------|-----------------|---------|
| 0 | - | - | - | - | 10000 | 2048 | **9.44** | **12.754** | 0.52 | pass |
| 7 | - | - | - | - | 10000 | 4096 | **8.68** | **12.969** | 0.56 | pass |
| 3 | - | - | 0.72-0.88 | - | 10000 | 2048 | **3.75** | **12.756** | 0.56 | pass |
| 1 | 0.02 | - | - | - | 10000 | 2048 | **4.11** | 7.016 | 0.86 | degraded, 55% |
| 2 | - | +-pi | - | - | 10000 | 2048 | 0.13 | 0.992 | 0.45 | **fail** |
| 4 | - | - | - | 0.1 rad | 10000 | 2048 | 0.14 | 0.232 | 0.24 | **fail** |
| 5 | 0.02 | +-pi | 0.72-0.88 | 0.1 rad | 10000 | 2048 | 0.12 | 0.121 | 0.07 | **fail** |
| 6 | 0.02 | +-pi | 0.72-0.88 | 0.1 rad | 30000 | 2048 | 0.12 | 0.121 | **1.68** | **fail** |
| 8 | 0.02 | +-pi | 0.72-0.88 | **-** | 10000 | 2048 | 4.48@499 | **8.952** | 0.81 | **pass** |
| 9 | 0.02 | +-pi | 0.72-0.88 | **-** | 30000 | 2048 | 2.41@499 | **8.679** @2500 | 1.00 | **pass** |

### Findings

**Batch size is not involved.** try0 (2048) and try7 (4096) are within 2% of each other
and of the last good teacher. The `num_envs` reduction forced by `replicate_physics`
costs nothing here.

**Cube scale is free.** try3 matches baseline exactly (12.756 vs 12.754) despite
starting slower (3.75 vs 9.44 at iteration 399).

**Observation noise costs about 45% of the lift reward but trains.** That is the price
of the robustness it buys, not a failure.

**The two blockers are cube yaw and initial arm pose,** and the initial arm pose is the
worse of the two (0.232 vs 0.992) even though it is only +-0.1 rad, moving the TCP by
3-4 cm. This was not expected: the reasoning when it was added was that an initial pose
is fully observable through `joint_pos`, so the policy only has to adapt. That reasoning
was wrong.

A plausible mechanism, untested: the offset is applied to all 7 arm joints including the
wrist, so it rotates the gripper as well as moving it. Yaw randomisation and arm-init
randomisation may be failing for the same underlying reason -- both destroy a fixed
alignment between gripper and cube that early training otherwise exploits.

**The curriculum is an amplifier, not the cause.** The `action_rate` / `joint_vel`
penalties jump 1e-4 -> 1e-1 at `common_step_counter > 10000`, i.e. iteration 417 at
`num_steps_per_env = 24`, regardless of `num_envs`. Among tries 0-4 this looked like a
cliff: every try with `lift > 3` at iteration 399 recovered, every try with
`lift ~ 0.13` died, with nothing in between.

try6 refutes the simple version of that story. Delaying the ramp to iteration 1250 did
**not** rescue the all-on configuration -- but it did change the failure mode. try5
collapsed (noise std 0.07, reach fell to 0.36); try6 kept exploring (noise std rising to
1.68, reach holding at 0.63). So the curriculum turns slow learning into permanent
failure, while delaying it leaves the policy still searching. try6 was not stuck at
1000 iterations, merely unfinished.

Note that try0 only cleared the ramp because lifting happened to emerge at iteration
300-400. The last good teacher did the same. The margin was always thin.

### Resolution

**The initial arm pose was the blocker.** try8 is try5 with that one term removed and
nothing else changed: `lifting_object` goes from 0.121 to 8.952. Observation noise,
cube yaw and cube scale together train fine.

8.95 against the clean baseline's 12.75 is the price of the added difficulty, not a
failure -- the teacher is solving a harder task. `position_error` ends at 0.187
(baseline 0.161).

**The curriculum delay was not needed and is not kept.** try8 (ramp at 10000) and try9
(ramp at 30000) converge to the same place -- final reward 61.9 against 62.0 -- but
try8 gets there in 1000 iterations where try9 needs 2500. The ramp was only dangerous
while the task was too hard to clear it in time.

Adopted in `lift_env_cfg.py`: observation noise, cube yaw and cube scale on,
`reset_robot_joints` removed, `SMOOTHNESS_PENALTY_ONSET` back to 10000.

### What has not been established

- whether the wrist-rotation hypothesis for try4 is correct. The offset was applied to
  all 7 arm joints, so it rotates the gripper as well as moving it, which would make it
  fail for the same reason cube yaw does. Randomising only joints 1-4 would test it,
  and would be the way to reintroduce start-pose variety if it is wanted.
- whether try6 (all four, delayed ramp) eventually succeeds. It was still exploring
  when it was stopped -- noise std rising to 1.68, reach holding at 0.63 -- rather than
  collapsed like try5. Moot now that the blocker is identified.
- single runs per cell. unitree_rl_lab's sandbox lineage recorded PPO run-to-run
  variance large enough to flip verdicts; a shared seed removes seed choice as a
  difference between tries but not the variance itself. Treat the clean passes
  (12.754 / 12.969 / 12.756), the clear failures (0.232 / 0.121) and the try5-to-try8
  jump as solid; treat try1's 7.016 against try8's 8.952 as directional only, since
  try8 has strictly more randomisation yet scored higher.
