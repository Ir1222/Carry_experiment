# Box-borne Perturbation Curriculum Resume Task

## Scope

`carrybox_boxperturb_resume` inherits the nominal `carrybox_resume` environment and
PPO configuration. It preserves the carry goal, rewards, policy observations,
privileged observations, action space, AMP setup, and network architecture. The only
new physical input is one external force pulse applied at the free box rigid-body COM
after confirmed bimanual carrying has remained stable for 20 policy steps.

Compatibility invariants:

- Actor observation: 738
- Critic observation: 143
- Interaction privileged tail: 17
- Actor and critic network definitions: inherited unchanged
- Existing reward scales and PPO hyperparameters: inherited unchanged
- `goal_pos`: unchanged; external force is not a command or observation

The legacy generic disturbance is disabled only in this task with
`domain_rand.disturbance = False`. Other robot domain randomization remains enabled.
Box size is fixed at `[0.3, 0.3, 0.25]` m and density at `50.0 kg/m^3` for this first
version.

`box_perturbation.enabled` is the master gate. Set it to `False` to run the same
task and policy with nominal box dynamics. Turning it off also clears a pending pulse
or recovery window immediately, preventing force leakage if the value is changed at
runtime. `debug_force_event` is separate: it makes the one-time probability sample
succeed but only when the master gate is enabled.

Set `debug_draw_force = True` for viewer diagnostics. An orange arrow starts at the
box COM and shows the instantaneous commanded external force from the most recent
physics substep. Its length is `|F| * debug_force_draw_scale_m_per_N` (default
`0.08 m/N`). The console prints the world-frame vector, magnitude in newtons, and
scheduled peak. This arrow is the injected pulse, not Isaac Gym's net contact force.

For a deterministic one-robot stress test, use `--debug_force_sweep`. Each
`(direction, beta)` cell receives a fresh `carryWith` RSI reset; no pulse is allowed
until 20 consecutive confirmed-carry policy steps. The beta grid is
`(0.10, 0.25, 0.50, 0.75)`, giving nominal peaks of about
`1.10, 2.76, 5.52, 8.28 N` for the 1.125 kg box. Robot DR and observation noise are
disabled. A cell that cannot enter confirmed carry within 5 seconds is reported as
`precondition_failed` and receives no force. This mode is evaluation-only.

The terminal reports the external force plus left/right `hand_on_box_proxy` vectors.
The proxy is the negative of each collision-hand body's net contact force, matching
the raw signal used by the 17D interaction privileged critic tail. It is not a true
pairwise wrench; the one-env evaluator additionally audits PhysX hand-box contact
pairs and their normal `lambda` values. Add `--verbose_force_trace` to print all 20
physics-substep samples of each pulse.

## Event and force timing

The base `post_physics_step()` refreshes simulator tensors and updates
`confirmed_carry_buf`. The derived task then increments or clears
`confirmed_carry_streak`. At the first streak value of 20, it samples the active
stage's event probability exactly once for that episode. At most one event can be
scheduled.

The next `step()` applies the scheduled force before every `gym.simulate()` call in
the four-substep control decimation loop. A dedicated full rigid-body force tensor is
zeroed on every physics substep, populated only at the one-body box index, and passed
to `gym.apply_rigid_body_force_tensors(..., ENV_SPACE)`. This applies a world-axis
force at the box COM with no external torque. Velocities are never modified.

The 0.10 s half-sine pulse uses 20 physics steps at `sim.dt = 0.005` s. Each
piecewise-constant force is sampled at the center of its physics interval:

`F_k = F_peak sin(pi (k + 0.5) / pulse_steps)`

The direction is transformed from the box x/y axis to world coordinates once when
the event is scheduled, then frozen. Downward perturbations use `[0, 0, -1]` directly.

## Force magnitude and curriculum

For each environment, the runtime mass returned by
`get_actor_rigid_body_properties(...)[0].mass` is used:

`F_peak = min(beta * m_box * 9.81, 10.0 N)`

| Stage | Start policy step | Directions | Beta ranges | Probability |
|---|---:|---|---|---:|
| C1 | 0 | ±box_x | x: 0.05–0.10 | 0.25 |
| C2 | 250,000 | ±box_x | x: 0.10–0.20 | 0.40 |
| C3 | 750,000 | ±box_x, -z | x: 0.10–0.25; z: 0.05–0.15 | 0.50 |
| C4 | 1,500,000 | ±box_x, ±box_y, -z | x: 0.10–0.30; y: 0.05–0.15; z: 0.05–0.20 | 0.60 |

`manual_stage_override` can be set to `"C1"` through `"C4"` for evaluation.
`debug_force_event = True` forces the one-time probability decision to succeed while
retaining the selected stage's direction and beta sampling.

## Recovery and TensorBoard

Recovery starts after the pulse and lasts at most 1.0 s (50 policy steps). It succeeds
when `confirmed_carry_buf` is true for five consecutive policy steps. Resetting during
an active recovery counts as a failed recovery. All per-environment perturbation state,
including the force tensor and carry streak, is cleared on reset.

The runner writes these rollout metrics directly under `Perturb/`:

- `stage_id`
- `enabled`
- `event_trigger_rate`
- `events_per_episode`
- `force_peak_N_mean`
- `beta_mean`
- `mass_kg_mean`
- `force_cap_usage_rate`
- `recovery_success_rate`
- `active_pulse_fraction`
- `recovery_window_fraction`

Episode summaries also include `Episode/Perturb/events_per_episode` and
`Episode/Perturb/recovery_success_rate`.

## Training

Run from `legged_gym/legged_gym/scripts` in the Isaac Gym Python environment:

```bash
python train.py \
  --task carrybox_boxperturb_resume \
  --resume \
  --resume_path <ckpt> \
  --run_name boxperturb \
  --max_iterations 30000
```

The staged-policy-step counter begins at zero when this environment is constructed,
so a nominal carry checkpoint starts the new perturbation curriculum at C1.

## Evaluate the bundled pretrained policy

The bundled `carrybox.pt` has a compatible 738-D actor and a legacy 126-D critic.
`play.py` therefore loads only its actor for this evaluation task; the perturbation
environment's 143-D critic is not needed for inference.

Deterministic perturbation-on evaluation:

```bash
python play.py \
  --task carrybox_boxperturb_resume \
  --resume_path "{LEGGED_GYM_ROOT_DIR}/resources/ckpt/carrybox.pt" \
  --debug_force_event
```

Nominal A/B control with the identical task and actor but the perturbation gate off:

```bash
python play.py \
  --task carrybox_boxperturb_resume \
  --resume_path "{LEGGED_GYM_ROOT_DIR}/resources/ckpt/carrybox.pt" \
  --disable_box_perturb
```

## Fair pretrained-vs-interaction-privileged A/B evaluation

Run 5 paired seeds for every direction and beta:

```bash
python legged_gym/scripts/evaluate_carrybox_boxperturb.py \
  --baseline_checkpoint resources/ckpt/carrybox.pt \
  --interaction_checkpoint /path/to/model_41000.pt \
  --seeds 1 2 3 4 5 \
  --output_dir logs/boxperturb_ab
```

Both checkpoints load only their 738D Actor. The evaluation produces:

- `trials.csv`: one row per independent cell and seed;
- `force_trace.csv`: physics-substep force/contact trace;
- `summary.csv`: grouped robustness and hand-force statistics;
- `comparison.json`: paired interaction-minus-baseline differences.

Primary robustness outcomes are confirmed-carry retention, recovery within 1 second,
drop/fall rate, hand-box relative motion, balance, and 2-second post-pulse task
continuation. Hand-force magnitude is explanatory rather than a standalone success
criterion.

## Validation

Compile from the repository root:

```bash
python -m compileall legged_gym/legged_gym rsl_rl/rsl_rl
```

Run the 16-environment deterministic debug smoke test from
`legged_gym/legged_gym/scripts`:

```bash
python validate_carrybox_boxperturb.py --num-envs 16 --rl-device cuda:0
```

The smoke test checks task registration, fixed box settings, pulse/recovery lengths,
one-event gating, runtime-mass force computation, force isolation to the box body,
finite 738/143 observations, the 17D interaction privileged tail, unchanged network
input sizes, and the required perturbation log keys.
