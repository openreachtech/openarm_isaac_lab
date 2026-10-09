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


---

## Narrowing the cube's spawn box, and a seed that will not train (2026-10-07)

### Why

Measuring the teacher trained on the wide box (x 0.30-0.50, y +-0.25) over 4096
episodes showed grasping failed as a clean function of distance from the arm base:

    0.30-0.35 m   1.0%      0.45-0.50 m  36.7%
    0.35-0.40 m   1.9%      0.50-0.56 m  75.6%
    0.40-0.45 m   9.6%                   20.8% overall

Cube size and yaw had no effect (20-22% in every bin) and removing the observation
noise barely helped, so it was reach, not perception. The TCP reaches 0.616 m from the
base and 0.602 m horizontally at table height, but a top-down grasp needs more arm
folded than that; the practical limit is about 0.45 m, 15 cm short of the kinematic one.

### What changed

Spawn box narrowed to x in [0.30, 0.42], y in [-0.18, 0.18], far corner 0.559 -> 0.457 m.

Separately, `noise_std_type` was switched from rsl_rl's default `"scalar"` to `"log"`.
Under `"scalar"` the action-noise std is a raw `nn.Parameter` handed to `Normal` as its
scale with nothing keeping it positive; one run crashed at iteration 364 with
"normal expects all elements of std >= 0.0". `"log"` parameterises it as `exp(log_std)`.
This was not only a crash fix: under `"log"` the std rose as high as 3.48 where
`"scalar"` peaked at 2.02 before collapsing, so exploration survives much longer.

### Result

| | wide box, "scalar" | narrow box, "log", seed 42 |
|---|---|---|
| lifting_object | 9.356 | **11.266** |
| object_goal_tracking_fine_grained | 0.542 | **1.403** |
| position_error | 0.146 | **0.104** |
| never picked up (4096 episodes) | 20.8% | **2.9%** |
| held at episode end | 78.8% | **95.4%** |
| final goal distance, successes | 6.7 cm | **3.3 cm** |

Failure is now flat across the box: 1.8% / 2.4% / 4.9% over the three distance bins,
with the worst cell the far corner (x 0.40, y +0.20) at 17%.

### The open problem: it trains about half the time

Same configuration, same iteration count, seed 7 instead of 42: `lifting_object` 0.127,
never grasped. The failure mode is consistent and distinct from the curriculum cliff --
`reaching_object` saturates around 0.63-0.71 while lifting stays at its random-policy
value, i.e. the gripper learns to hover a few centimetres from the cube and never
closes. Only once that has gone on long enough does the smoothness ramp finish it off.

Four narrow-box runs so far: seed 42 under `"scalar"` failed twice (once crushed by the
ramp at iteration 417, once at 1250), seed 7 crashed on the std bug, and after the
`"log"` fix seed 42 succeeded and seed 7 did not. So the honest count after the fix is
one success in two.

A single failed run is therefore not evidence that a configuration is wrong. Anything
compared from here needs more than one seed.

The untested fix for the hovering optimum is the shaping term itself: `reaching_object`
pays `1.1 * (1 - tanh(d / 0.1))`, which at the 3.3 cm hover distance is still worth
about 0.7 of its maximum. Tightening the kernel or cutting the weight would stop
hovering from paying, at the risk of weakening the guidance that gets the arm there.


---

## Where the trembling came from: the arm base height (2026-10-08)

### Why

The teacher held the cube but shook while doing it, in about a fifth of episodes, worst
when the goal sat close to the base (45% under 0.32 m against 7% in the middle). Tracing
one episode showed the commanded joint angle for joint7 asking for 9.8 rad against a
+-1.571 limit -- the policy was driving far outside the reachable range, where every
command has the same effect and the reward is flat, so it drifted. The actual joint was
*smoother* than the target, which ruled out the actuator.

### What was tried first, and why it was wrong

Three changes aimed at that diagnosis, all since reverted:

| change | result |
|---|---|
| action normalised so -1/+1 spans each joint's range | joint velocity while holding 0.408 -> 0.973; trembling spread from 20% of episodes to 98% |
| `action_out_of_range_l2`, penalising only commands past the range | action magnitude 7.4 -> 1.7, no effect on the trembling |
| `joint_target_rate_l2`, smoothness measured in radians rather than action units | at weight -1.0 the trembling nearly vanished (0.094) but `fine_grained` fell to 0.906 and one seed in two unlearned its grasp entirely; at -0.2 it was too weak to help |

Worse, under a strong rate penalty the policy found that the cheapest way to hold still
is to **push into a joint stop**: joint2 sat at exactly 1.745 in all 972 held episodes,
standard deviation 0.000, which is the "every arm has the same odd shape" the behaviour
showed on screen. That is the unnatural-but-works-in-sim failure DextrAH-G warns about
(section 3.2, figures 10-12). It also appeared at weight -0.2 on one seed and not the
other, so it was never a matter of tuning the weight.

Two things found along the way were real and were kept: `noise_std_type` must be `"log"`
(the default `"scalar"` feeds an unconstrained parameter to `Normal` as its scale and one
run crashed with "normal expects all elements of std >= 0.0"), and `action_rate_l2`
measures smoothness in action units, so its meaning silently changes with the scaling.

### The actual cause

Restoring the first-commit task as ``Isaac-Lift-Cube-OpenArm-v0`` showed it holding the
cube cleanly and symmetrically. Sweeping only the base height over that otherwise
untouched task:

| base z | fine_grained | goal distance y<0 / y>0 | left-right gap |
|---|---|---|---|
| 0.00 | 3.820 | 0.36 / 0.35 cm | 0.01 cm |
| 0.05 | 3.699 | 0.21 / 0.23 cm | 0.02 cm |
| 0.10 | 3.499 | 0.29 / 0.41 cm | 0.12 cm |
| 0.15 | 1.566 | 3.66 / 5.41 cm | 1.75 cm |

Not a gradual cost but a cliff between 0.10 and 0.15, and the mechanism is visible in
the joint angles: at 0.05 both joint5 and joint7 still differ between left and right
goals, at 0.10 joint5 is pinned to its stop and only joint7 adjusts, and at 0.15 joint7
is pinned too. With nothing left to aim with, the arm leans the same way whichever side
the goal is on. joint6, whose travel is +-0.785 against +-1.571 for its neighbours, sits
on a stop at every height.

### Adopted

Base height 0.15 -> **0.10**, the action-space changes reverted, everything else kept.
Two seeds, 2500 iterations, with all of the randomisation on:

| | lifting_object | fine_grained | position_error |
|---|---|---|---|
| seed 42 | 11.78 | 0.678 | 0.097 |
| seed 7 | 11.59 | 1.343 | 0.107 |

against 11.27 / 1.403 / 0.104 at the old 0.15. Trembling and the one-sided posture are
both gone on inspection.

That also separates the two costs at last. Base height 0 -> 0.10 costs 8% of
`fine_grained` (3.820 -> 3.499); the randomisation -- observation noise above all --
costs the remaining 62-81% (3.499 -> 0.68-1.34). The noise is a deliberate trade: half
of its sigma = 0.02 is a bias held for the whole episode, against a reward kernel 5 cm
wide, and it buys a teacher that still works when a depth student's estimate is a few
centimetres off.

Note the spread between the two seeds: `fine_grained` 0.678 against 1.343, a factor of
two, while `lifting_object` and `position_error` barely move. Grasping and carrying are
stable; only the last few centimetres of placement swing with the seed.
