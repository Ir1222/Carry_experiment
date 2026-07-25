# PhysHSI CarryBox deployment

This package deploys the deterministic actor stored in
`legged_gym/logs/Jul09_from_55500/model_73500.pt`. The checkpoint is an
`rsl_rl` state dictionary, not a standalone Python model. The exporter extracts
only the actor (`738 -> 512 -> 256 -> 256 -> 29`) and produces the ONNX artifact
used by the policy process.

## What is implemented

- Exact 123-D frame / 6-frame history actor observation.
- Explicit 29-joint and five-endpoint name mapping.
- PhysHSI position-target action scaling and PD control.
- Versioned UDP task-state protocol for box/goal perception.
- Local UDP sim2sim transport for development and tests.
- FALCON-style Unitree DDS `LowState` / `LowCmd` dual-process transport.
- A fail-closed Unitree sim2real skeleton; hardware writes are disabled by
  default.

FALCON is only a structural reference. Its G1 model is not used because its
torso mass, waist/shoulder geometry, effort limits, endpoint bodies, and some
configuration labels differ from the PhysHSI training asset.

## Ubuntu setup

Use Python 3.10 on Ubuntu 22.04:

```bash
python -m pip install -r deploy/requirements.txt
python -m deploy.tools.build_mjcf
python -m deploy.tools.export_actor
```

For DDS mode, install `unitree_sdk2_python` from Unitree's repository. The
generated robot MJCF and exported ONNX are deliberately ignored by Git: the
MJCF contains machine-local absolute mesh paths and the ONNX contains model
weights.

`joint_armature: 0.01` is carried over from the PhysHSI training asset options;
without it, the high-gain wrist dynamics are numerically unstable at 200 Hz.

## Run complete sim2sim

DDS mode mirrors the real G1 topics. Start the simulator first:

```bash
python -m deploy.sim2sim.mujoco_server \
  --config deploy/config/g1_carrybox.yaml \
  --transport unitree_dds
```

Then start and arm the policy:

```bash
python -m deploy.policy.run \
  --config deploy/config/g1_carrybox.yaml \
  --mode sim2sim \
  --transport unitree_dds \
  --arm
```

For development without Unitree SDK, use `--transport udp` in both commands.
The MuJoCo window accepts Backspace to reset and Q/Escape to quit. The policy
terminal accepts these commands followed by Enter:

- `]` or `arm`: arm policy.
- `o` or `hold`: current-position damping hold.
- `x` or `estop`: latch emergency stop.
- `clear`: clear e-stop, remaining disarmed.
- `q`: quit.

The simulator runs at 200 Hz and policy inference at 50 Hz. The server holds
each position target for four physics steps and clips computed torque to the
effort limits parsed from the PhysHSI URDF.

The deterministic reset is frozen until the first armed policy command. This
keeps the open-loop humanoid upright while ONNX Runtime and DDS initialize;
after activation, normal free-base dynamics and stale-command fallback apply.
Optional `--log path.jsonl` on both executables records observation slices,
actions, targets, gains, torque, task pose, and contact counts. `--duration`
provides a bounded headless smoke run.

## Sim2real dry-run

The actor needs box pose, box size, and goal pose relative to the torso. These
signals do not exist in Unitree `LowState`; a perception process must publish
the versioned task-state UDP packet. For interface testing only:

```bash
python -m deploy.sim2real.mock_task
python -m deploy.policy.run --mode sim2real --arm
```

`safety.dry_run: true` means the policy runs and logs actions but the Unitree
backend never writes `LowCmd`. Real writes require both:

1. changing `safety.dry_run` to `false` in a commissioning copy of the YAML;
2. passing `--allow-hardware-command`.

The initial hardware `kp_scale` is zero. Increase it only in a reviewed
commissioning configuration after validating IMU frame, motor mapping, joint
limits, estimator/task-state latency, and current-position fallback. Stale
robot/task state, non-finite values, invalid quaternion, excessive tilt,
joint-limit violations, or e-stop all return to a damping hold and require an
explicit re-arm.

When `robot.imu_frame: pelvis` is selected, the backend uses waist FK to
transform orientation and gyro into `torso_link`; the default `torso` setting
expects Unitree `LowState.imu_state` to already represent the torso frame.
Hardware stiffness is ramped over `safety.gain_ramp_seconds`, and the 500 Hz
command thread independently replaces a stale policy target with a zero-Kp
current-position damping hold.

## Observation contract

One frame is concatenated in this exact order:

1. torso angular velocity x 0.25 (3)
2. projected gravity in torso frame (3)
3. joint position minus default (29)
4. joint velocity x 0.05 (29)
5. left palm, right palm, left ankle-pitch, right ankle-pitch, and head
   positions (15)
6. previous raw actor action (29)
7. torso-relative box position, box X/Z rotation axes, box size, and goal
   position (15)

The oldest-to-newest six-frame history is 738-D. The default compatibility
profile preserves the one-policy-step ankle-position delay present in the
training environment. Synthetic deployment noise is disabled.

## Verification

Run:

```bash
python -m pytest tests/test_deploy_core.py -q
```

The suite checks the state-dict actor contract, ONNX parity, quaternion and
history ordering, legacy ankle delay, named URDF/MuJoCo/actuator mappings,
PD/effort clipping, UDP packet round trips, fail-closed inference behavior,
and a 10-second MuJoCo finite-state smoke test.

## Known limits

- The first scene uses a deterministic box/goal reset and does not reproduce
  AMP motion-state initialization, curriculum, or all PhysX randomization.
- PhysX and MuJoCo contact solvers differ; observation/action parity does not
  guarantee task-level cross-simulator success.
- The supplied checkpoint can still trigger the safety fallback in the fixed
  MuJoCo scene; task-level tuning is deliberately separate from interface,
  mapping, and numerical-parity validation.
- Real task perception and global localization are not implemented.
- The configured Unitree IMU frame and motor mapping must still be confirmed on
  the target G1 before changing `dry_run`.
