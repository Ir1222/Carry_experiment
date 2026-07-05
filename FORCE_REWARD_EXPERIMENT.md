# Box-perturbation force-quality reward experiment

## Scope and interfaces

Task: `carrybox_boxperturb_force_reward_resume`.

The task adds simulator-side reward shaping and diagnostics only. Actor observations remain 738D, Critic observations remain 143D, and the interaction privileged tail remains 17D. No force, perturbation vector, or direction identifier is added to either policy input. Network architecture and PPO settings are inherited unchanged from `carrybox_resume`.

In this checkout the previously described `carrybox_boxperturb_resume` implementation is not present. The force layer therefore exposes `box_perturb_active_buf`, `perturb_recovery_window_buf`, and `box_perturb_direction_id_buf` as its event interface. Normal runs must connect these buffers to the completed perturbation scheduler. `force_reward.debug_force_c1=True` provides only a deterministic +x box-COM diagnostic pulse; it is not a replacement for the C1-C4 curriculum.

## Force source and interpretation

`carrybox.py::_init_buffers()` acquires `gym.acquire_net_contact_force_tensor(sim)` and views it as `[num_envs, 3 + num_robot_bodies, 3]`. Collision-hand indices come from `find_actor_rigid_body_handle`; the box index comes from `get_actor_rigid_body_handle`. `box_states` is the box actor root state, collision-hand positions are in `rigid_body_states`, box dimensions are `_box_size`, mass is `box_masses`, and the carry gate is `confirmed_carry_buf`.

The reward layer reads these raw world-frame rigid-body net-contact resultants directly. It does not reuse the existing Critic-side base-frame vectors. They are not exact contact-point forces, normals, pairwise wrenches, or friction-utilization measurements.

At the first stable confirmed-carry sample, each collision-hand position is transformed into box coordinates. The axis with maximum `abs(r_hand_box) / half_extent` selects a face, and the coordinate sign selects its outward normal. The face is locked until reset. During the initial verification interval, mean `dot(F_hand_world, n_world)` selects a once-per-episode normal sign. Environments with persistently low positive-normal fraction are warned and permanently disabled until reset.

At policy rate, with `alpha = exp(-(sim.dt * decimation) / 0.04)`, normal and tangential magnitudes receive EMA filtering. The implemented quantities are:

- `F_n_signed = dot(F_hand_world, n_world)`, `F_n = relu(F_n_signed)`
- `F_t = norm(F_hand_world - F_n * n_world)`
- `f_n = EMA(F_n) / (box_mass * 9.81 + eps)`
- `rho = clamp(EMA(F_t) / (EMA(F_n) + eps), 0, rho_max)`
- `closure_residual = norm(F_left + F_right + F_box) / (norm(F_left) + norm(F_right) + norm(F_box) + eps)`
- `normal_load_asymmetry = abs(F_n_left - F_n_right) / (F_n_left + F_n_right + eps)` (diagnostic only)

Exactly three registered rewards are added. Normal margin and shear-risk terms use confirmed perturb-active or recovery gates. Over-compression uses confirmed carry. All also require valid sign verification. Huber outputs and all raw reward outputs are clamped/bounded to `[0, 1]`.

## Calibration

Defaults are `force_reward.enabled=False` and `force_reward.calibration_only=True`; therefore configured reward scales do not affect return during the first diagnostic run. Placeholder dimensionless thresholds are inactive scientific defaults and must be replaced from rollout distributions:

- `f_n_min = 0.8 * Q20(f_n)`
- `f_n_sat = Q70(f_n)`
- `f_n_max = 1.25 * Q95(f_n)`
- `rho_safe = 1.10 * Q90(rho)`

The configured scales are `normal_force_margin=0.06`, `tangential_compression_risk=-0.08`, and `normal_force_overcompression=-0.02`. After calibration, set `enabled=True` and `calibration_only=False`.

TensorBoard receives `Force/` tags for means/p95 (and closure maximum), including unprefixed all-confirmed statistics and `confirmed/`, `nominal/`, `perturb_active/`, `recovery/`, and `direction_N/` groups. Required diagnostics include force/weight ratios, rho, load asymmetry, positive-normal fractions, closure residual, six face-ID bins per hand, valid fraction, and each raw reward. Successful-episode confirmed-carry means are emitted under `Episode/Force/successful_confirmed/`.

## Ablation

- B0: perturbation enabled, force-quality rewards disabled, Critic interaction PI disabled.
- B1: perturbation enabled, force-quality rewards enabled, Critic interaction PI disabled.
- B2: perturbation enabled, force-quality rewards enabled, Critic interaction PI enabled.

B0 vs B1 measures force-aware reward shaping. B1 vs B2 measures the additional effect of Critic-side interaction privileged information. B2 alone does not isolate Critic PI from force supervision.

## Commands

Calibration training:

```powershell
python legged_gym/legged_gym/scripts/train.py --task carrybox_boxperturb_force_reward_resume --headless
```

Runtime smoke test without checkpoint loading:

```powershell
python legged_gym/legged_gym/scripts/validate_carrybox_force_reward.py --headless
```

Optional shape-compatible resume after supplying a checkpoint path:

```powershell
python legged_gym/legged_gym/scripts/train.py --task carrybox_boxperturb_force_reward_resume --resume --resume_path D:\path\to\model.pt --headless
```

Deterministic C1 diagnostic with vector visualization/logging:

```powershell
$env:FORCE_DEBUG_C1='1'
python legged_gym/legged_gym/scripts/validate_carrybox_force_reward.py
```

## Limitations

- Net rigid-body resultants can include contacts other than the box. Closure residual and viewer inspection must establish whether the selected collision-hand bodies contact only the box during confirmed carry.
- The nearest-face proxy is unreliable for corner/edge contacts and cannot recover a true contact normal or friction coefficient.
- EMA values depend on policy rate and simulator contact collection.
- Smoke tests establish integration and finiteness only; they do not establish reward effectiveness.
- This checkout cannot support a normal C1-C4 experiment until the missing perturbation scheduler drives the three event-interface buffers.
