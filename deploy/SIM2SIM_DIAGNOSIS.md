# CarryBox Sim2Sim diagnosis

## Current result

The shared deployment path is no longer the cause of both actors failing.
The strict 20-second dual-process UDP validation on 2026-07-29 produced:

| Profile | Result | Policy / physics | p99 inference | Max tilt XY | Max limit penetration | First failure |
|---|---:|---:|---:|---:|---:|---|
| `official_carrybox_65000` | PASS | 50.01 / 199.76 Hz | 0.63 ms | 0.358 | 0.0116 rad | none |
| `model_73500` | FAIL | 50.03 / 199.91 Hz | 0.49 ms | 0.733 | 0.0013 rad | both hip-yaw links touch the ground at about 11.1 s |

The second result is a task/trajectory parity failure, not a transport,
manifest, joint mapping, action-decimation or inference-rate failure.
`model_73500` remains armed and numerically finite, but converges to an
extremely low crouch in the deterministic MuJoCo scene.

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
