# CarryBox Sim2Sim diagnosis

## Current result

The shared deployment path is no longer the cause of both actors failing.
The strict 20-second dual-process UDP validation on 2026-07-29 produced:

| Profile | Result | Policy / physics | p99 inference | Max tilt XY | Max limit penetration | First failure |
|---|---:|---:|---:|---:|---:|---|
| `official_carrybox_65000` | PASS | 50.01 / 199.76 Hz | 0.63 ms | 0.358 | 0.0116 rad | none |
| `model_55500` | FAIL | 49.94 / 199.85 Hz before termination | 0.48 ms | 0.620 | 0.0052 rad | training roll termination at about 11.4 s |
| `model_73500` | FAIL | 50.03 / 199.91 Hz | 0.49 ms | 0.733 | 0.0013 rad | both hip-yaw links touch the ground at about 11.1 s |

Both locally trained policies fail at almost the same task time, but through
different terminal modes. `model_55500` rolls beyond the Isaac termination
threshold (`roll=-0.514 rad`); `model_73500` remains numerically finite but
converges to an extremely low crouch and puts both hip-yaw links on the
ground. These are task/trajectory parity failures, not transport, manifest,
joint mapping, action-decimation or inference-rate failures.

At failure, neither policy has lifted the box. `model_55500` has moved the box
only about `[+0.058, -0.157, 0.000] m`, while `model_73500` has moved it about
`[+0.370, -0.280, 0.000] m`. The official actor stays upright for 20 seconds
under the same deployment path, which rules out a deployment defect that
would make every actor unstable.

## Training evidence

The TensorBoard histories show that continued optimization improved aggregate
reward without improving the measured carry interaction:

- Mean reward rises from about `37.4` near iteration 55,500 to `49.2` near
  iteration 73,500.
- Confirmed-carry ratio falls from about `0.0525` to `0.0413`.
- Both-hand contact ratio falls from about `0.1235` to `0.1055`.
- Lifted bimanual contact ratio falls from about `0.469` to `0.400`.
- The magnitude of the joint-limit penalty worsens from about `0.106` to
  `0.177`.

The current training configuration also has long-range carry progress,
carry-stability and success-termination rewards disabled. Hybrid reset uses a
reference-motion initialization 80% of the time, while the deterministic
MuJoCo test starts from a complete approach/pickup state. Consequently, higher
PPO return is not evidence of a more stable end-to-end carry policy.

The MuJoCo scene still does not reproduce every training task detail: Isaac
uses phase-specific motion states, randomized box properties and source/target
platforms. Those task/contact gaps must be isolated with deterministic Isaac
phase snapshots before assigning the entire failure to physics transfer.

## Confirmed deployment defects that were fixed

- Actor observation is now constructed in the trained `pelvis` policy frame,
  not `torso_link`.
- Robot and task UDP states are paired only when their source sequence is
  identical.
- Duplicate sequence packets are skipped and do not clear the six-frame
  observation history. Only a true sequence rollback resets history.
- UDP protocol version 2 rejects old torso-semantics packets.
- The server applies a 0.01 m collision margin and explicit joint-limit
  `solref`/`solimp` parameters.
- Position targets change only on a four-physics-step policy boundary.
- Sim-parity episode failure is separated from hardware fail-closed behavior.
  The simulator pauses/latches an Isaac-style termination instead of allowing
  a zero-Kp fallback to fall under gravity.
- The 200 Hz deadline scheduler no longer accumulates per-tick sleep error.
- Model profiles have independent ONNX/manifest paths and mandatory SHA256
  validation.
- The validator requires continuous validity and checks falls, unexpected
  ground contact, limits, sequence, control decimation and latency.

## What still requires the Isaac Gym lab host

Run the golden snapshot exporter/comparator with a deliberately nonzero waist
pose. The raw Isaac state and reconstructed deployment observation must match
at `1e-5`. This is the remaining authoritative check against the live training
environment.

Task-level parity also requires exporting deterministic Isaac initial states
for approach (`loco`), pickup, carry and putdown. Until those states and their
Isaac baselines are replayed in MuJoCo, the current validator proves the
deployment/control interface and upright stability of the official actor, but
does not claim an 80% four-phase carry success rate.

Do not hide a failing phase by reducing action scale, Kp or actor output.
Compare its first divergent observation/action/contact trace against the
matching Isaac golden trace.
