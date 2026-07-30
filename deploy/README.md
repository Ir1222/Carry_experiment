# PhysHSI CarryBox deployment

This package deploys either the official `carrybox.pt` actor or the trained
`model_73500.pt` actor. Both checkpoints are `rsl_rl` state dictionaries, not
standalone Python models. The exporter extracts only the deterministic actor
(`738 -> 512 -> 256 -> 256 -> 29`) into a profile-specific ONNX artifact.

## What is implemented

- Exact 123-D frame / 6-frame history actor observation.
- Explicit 29-joint and five-endpoint name mapping.
- PhysHSI position-target action scaling and PD control.
- Version-2 UDP packets whose robot and task states share one source sequence.
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
conda activate carryDeploy
cd ~/pyx/Carry_experiment

python -m pip install -r deploy/requirements.txt
python -m deploy.tools.build_mjcf
python -m deploy.tools.export_actor \
  --checkpoint legged_gym/resources/ckpt/carrybox.pt \
  --profile official_carrybox_65000

python -m deploy.tools.export_actor \
  --checkpoint legged_gym/logs/Jul09_from_55500/model_73500.pt \
  --profile model_73500

python -m deploy.tools.preflight \
  --mode udp-sim2sim \
  --profile model_73500
```

The preflight is read-only. It verifies Python/dependencies, required files,
manifest/checkpoint/ONNX hashes and interface, URDF mapping, MuJoCo actuator
count, 200/50/500 Hz rates, and free UDP ports before a process is started.
An export cannot silently overwrite the other named model profile.

`joint_armature: 0.01` is carried over from the PhysHSI training asset options;
without it, the high-gain wrist dynamics are numerically unstable at 200 Hz.

## Strict UDP Sim2sim validation (start here)

This mode does not import or require Unitree SDK2. Validate both named models
with one command; the validator starts and stops both processes itself:

```bash
python -m deploy.tools.validate_sim2sim \
  --config deploy/config/g1_carrybox.yaml \
  --models official_carrybox_65000,model_73500 \
  --duration 20 \
  --headless \
  --report-dir ~/physhsi_deploy_logs
```

Each profile gets independent policy/MuJoCo JSONL logs and a validation summary.
Passing requires continuous safe/armed control after warm-up, 50 Hz policy,
200 Hz physics, p99 inference below 15 ms, no history reset on duplicate
packets, no target changes inside the four-step interval, joint-limit
penetration below 0.02 rad, no training tilt termination, and no
pelvis/torso/hip ground contact.

For an interactive run, start MuJoCo in terminal 1:

```bash
python -m deploy.sim2sim.mujoco_server \
  --config deploy/config/g1_carrybox.yaml \
  --transport udp \
  --log ~/physhsi_deploy_logs/mujoco_udp.jsonl
```

Add `--headless` over SSH. Start the policy in terminal 2:

```bash
python -m deploy.policy.run \
  --config deploy/config/g1_carrybox.yaml \
  --mode sim2sim \
  --transport udp \
  --profile model_73500 \
  --arm \
  --log ~/physhsi_deploy_logs/policy_udp.jsonl
```

### D455 first-person camera

The generated robot MJCF contains a massless, collision-free
`d455_camera` attached to the trained `d455_link` optical frame. Rebuild once
after updating the deployment package:

```bash
python -m deploy.tools.build_mjcf
```

Start directly in the D455 view:

```bash
python -m deploy.sim2sim.mujoco_server \
  --config deploy/config/g1_carrybox.yaml \
  --transport udp \
  --camera-view d455 \
  --log ~/physhsi_deploy_logs/d455_mujoco.jsonl
```

Press `C` to toggle between the free third-person camera and D455. The camera
uses the repository's D455 extrinsics and an 848x480 depth-style field of view
(57.5 degree vertical, 88.21 degree derived horizontal). It is visualization
only and does not enter the actor observation.

At every 50 Hz policy boundary the MuJoCo log records `box_center_uv`,
`box_bbox_uv`, `center_depth_m`, `partially_visible`, `fully_visible`, and
`behind_camera`. Pixel coordinates always refer to the fixed 848x480
intrinsics, independent of the desktop window size.

`--arm` is the safety permission to send the inferred joint targets; it is
independent of the task's box start and goal. The automatic validator supplies
this permission after it receives valid synchronized robot/task state.

## Install Unitree SDK2

Use Unitree's official Python SDK source. The preflight only imports SDK
symbols and never creates a DDS writer:

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
python -m pip install -e .

python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_; print('Unitree SDK2 ready')"
```

If installation reports `Could not locate cyclonedds`, build the supported
0.10.x line and retry:

```bash
sudo apt update
sudo apt install -y git cmake build-essential

cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds \
  -b releases/0.10.x
cmake -S ~/cyclonedds -B ~/cyclonedds/build \
  -DCMAKE_INSTALL_PREFIX=~/cyclonedds/install
cmake --build ~/cyclonedds/build --target install -j"$(nproc)"
export CYCLONEDDS_HOME=~/cyclonedds/install

cd ~/unitree_sdk2_python
python -m pip install -e .
conda env config vars set CYCLONEDDS_HOME="$HOME/cyclonedds/install"
conda deactivate
conda activate carryDeploy
```

## DDS Sim2sim

DDS mode mirrors the real G1 topics. Start the simulator first:

```bash
python -m deploy.tools.preflight --mode dds-sim2sim

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
  --arm \
  --log ~/physhsi_deploy_logs/policy_dds.jsonl
```

Keep `network.interface: lo` and `domain_id: 0` when both processes run on the
same host. Box and goal task state still use UDP `127.0.0.1:15001`.

The MuJoCo window accepts Backspace to reset, C to toggle free/D455 view, and
Q/Escape to quit. The policy
terminal accepts these commands followed by Enter:

- `]` or `arm`: arm policy.
- `o` or `hold`: current-position damping hold.
- `x` or `estop`: latch emergency stop.
- `clear`: clear e-stop, remaining disarmed.
- `q`: quit.

The simulator runs at 200 Hz and policy inference at 50 Hz. The server holds
each position target for four physics steps and clips computed torque to the
effort limits parsed from the PhysHSI URDF.

The parity physics profile applies a 0.01 m collision margin only to collidable
robot/ground/box geoms, explicit joint-limit solver parameters to all 29
limited joints, and 0.01 linear/angular free-base damping. Startup prints and
logs the complete solver, contact, mass, inertia, friction and timestep
fingerprint.

The deterministic reset places the nominal 0.30 x 0.30 x 0.25 m box on a
fixed 0.40 x 0.40 x 0.02 m source platform. The box-center height is 0.335 m,
the median representative of Isaac Gym play's default `U(0, 0.65)` height
sample followed by its 0.01 m lift. The configured 0.02 m geometric gap
matches the two 0.01 m contact offsets/margins. This reproduces a typical
play-default elevated pickup, not the full stochastic or motion-reference
training reset.

The deterministic reset is frozen until the first armed policy command. At
each 50 Hz boundary the simulator publishes one state sequence and waits for
the command computed from that exact sequence. Repeated UDP packets do not run
inference or clear history; a true sequence rollback clears history once.
Sim-parity failures pause the episode instead of applying a zero-Kp fallback
and then reporting the gravity-driven fall as actor behavior.
Optional `--log path.jsonl` on both executables records observation slices,
actions, targets, gains, torque, task pose, and contact counts. `--duration`
provides a bounded headless smoke run.

## Sim2real dry-run

The actor needs box pose, box size, and goal pose relative to the pelvis policy
frame. These signals do not exist in Unitree `LowState`; a perception process
must publish the version-2 task-state UDP packet.

First identify the dedicated wired interface and create a temporary
safety-locked configuration. This does not modify the project YAML:

```bash
ip -br link
ip -br addr
export G1_NIC=enp3s0  # replace this value

python -m deploy.tools.make_dryrun_config \
  --interface "$G1_NIC" \
  --output /tmp/g1_carrybox_dryrun.yaml

python -m deploy.tools.preflight \
  --config /tmp/g1_carrybox_dryrun.yaml \
  --mode sim2real-dryrun
```

Terminal 1 publishes stationary mock perception data:

```bash
python -m deploy.sim2real.mock_task \
  --config /tmp/g1_carrybox_dryrun.yaml \
  --rate 50 \
  --box-pos 1.0 0.0 -0.65 \
  --goal-pos 2.5 0.75 -0.65
```

Terminal 2 receives real LowState and runs inference:

```bash
python -m deploy.policy.run \
  --config /tmp/g1_carrybox_dryrun.yaml \
  --mode sim2real \
  --transport unitree_dds \
  --arm \
  --log ~/physhsi_deploy_logs/g1_real_dryrun.jsonl
```

Do not add `--allow-hardware-command`. The generated file always sets
`safety.dry_run: true` and `hardware_kp_scale: 0.0`; the preflight refuses a
loopback interface and verifies both locks. The Unitree backend receives
LowState and executes ONNX, but never calls the DDS `LowCmd` writer.

Future real writes would require both:

1. changing `safety.dry_run` to `false` in a commissioning copy of the YAML;
2. passing `--allow-hardware-command`.

Neither action is part of this dry-run stage. Increase gain only in a reviewed
commissioning configuration after validating IMU frame, motor mapping, joint
limits, estimator/task-state latency, and current-position fallback. Stale
robot/task state, non-finite values, invalid quaternion, excessive tilt,
joint-limit violations, or e-stop all return to a damping hold and require an
explicit re-arm.

The trained policy frame is always pelvis. When Unitree LowState provides a
torso IMU (`robot.imu_frame: torso`), the backend uses waist FK to transform
orientation and gyro to pelvis. With `robot.imu_frame: pelvis`, LowState is
already in the policy frame.
Hardware stiffness is ramped over `safety.gain_ramp_seconds`, and the 500 Hz
command thread independently replaces a stale policy target with a zero-Kp
current-position damping hold.

## Observation contract

One frame is concatenated in this exact order:

1. pelvis angular velocity x 0.25 (3)
2. projected gravity in pelvis frame (3)
3. joint position minus default (29)
4. joint velocity x 0.05 (29)
5. left palm, right palm, left ankle-pitch, right ankle-pitch, and head
   positions (15)
6. previous raw actor action (29)
7. pelvis-relative box position, box X/Z rotation axes, box size, and goal
   position (15)

Endpoint positions are world positions minus the pelvis/root position and then
rotated into pelvis. The oldest-to-newest six-frame history is 738-D. The
default compatibility profile preserves the one-policy-step ankle-position
delay present in the training environment. Synthetic deployment noise is
disabled.

## Verification

Run:

```bash
python -m pytest tests/test_deploy_core.py tests/test_deploy_tools.py -q
python -m deploy.tools.preflight \
  --mode udp-sim2sim \
  --profile official_carrybox_65000
python -m deploy.tools.preflight \
  --mode udp-sim2sim \
  --profile model_73500
python -m deploy.tools.validate_sim2sim \
  --models official_carrybox_65000,model_73500 \
  --duration 20 \
  --headless \
  --report-dir ~/physhsi_deploy_logs
```

The suite checks the state-dict actor contract, ONNX parity, quaternion and
history ordering, legacy ankle delay, named URDF/MuJoCo/actuator mappings,
PD/effort clipping, version rejection, synchronized UDP state pairing,
fail-closed inference behavior, physics profile and a 10-second finite-state
MuJoCo test.

To produce and compare one golden Isaac Gym observation on the lab machine:

```bash
cd ~/pyx/Carry_experiment
python legged_gym/legged_gym/scripts/play.py \
  --task carrybox \
  --headless \
  --num_envs 1 \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --deploy_snapshot ~/physhsi_deploy_logs/isaac_snapshot.npz

python -m deploy.tools.compare_observation_snapshot \
  ~/physhsi_deploy_logs/isaac_snapshot.npz
```

The comparator independently reconstructs the 123-D newest frame from raw
Isaac world states and checks both it and the 738-D history at `1e-5`. It also
reports the pelvis/torso quaternion difference so a nonzero-waist snapshot
detects any future frame regression.

Export 20 deterministic reset samples for each task phase with:

```bash
for PHASE in loco pickUp carryWith putDown; do
  python legged_gym/legged_gym/scripts/play.py \
    --task carrybox \
    --headless \
    --num_envs 1 \
    --resume_path legged_gym/resources/ckpt/carrybox.pt \
    --deploy_snapshot \
      "$HOME/physhsi_deploy_logs/isaac_states/${PHASE}.npz" \
    --deploy_snapshot_phase "$PHASE" \
    --deploy_snapshot_count 20
done
```

When count is greater than one the exporter appends the phase and a
zero-padded index to each file. Every snapshot includes root/pelvis/torso pose
and velocities, joints, endpoints, box/goal, previous action, current 123-D
frame and the complete 738-D history.

## Known limits

- The first scene uses a deterministic play-default representative with a
  source platform (`[1.75, 0, -0.465]` box position approximately in pelvis
  coordinates). It does not reproduce AMP motion-state initialization,
  curriculum, the full box-height distribution, or all PhysX randomization.
- PhysX and MuJoCo contact solvers differ; observation/action parity does not
  guarantee task-level cross-simulator success.
- Approach/pickup/carry/putdown success rates still require the planned batch
  of phase-specific Isaac initial-state exports. A finite, upright interface
  run is not labeled as a completed carry task.
- Real task perception and global localization are not implemented.
- The configured Unitree IMU frame and motor mapping must still be confirmed on
  the target G1 before changing `dry_run`.
