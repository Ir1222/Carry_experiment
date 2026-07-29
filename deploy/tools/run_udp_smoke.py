"""Run and strictly verify one headless UDP MuJoCo/policy test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from deploy.common.config import load_deploy_config


FORBIDDEN_GROUND_BODY_TOKENS = ("pelvis", "torso", "hip")


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    model_profile: str
    policy_steps: int
    safe_steps: int
    safe_fraction: float
    policy_hz: float
    physics_steps: int
    physics_hz: float
    physics_wall_hz: float
    max_inference_ms: float
    inference_p99_ms: float
    max_projected_gravity_xy: float
    max_joint_limit_violation_rad: float
    forbidden_ground_contact_steps: int
    sequence_reset_count: int
    mid_interval_action_change_count: int
    non_decimated_policy_step_count: int
    first_failure_sequence: int | None
    first_failure_time: float | None
    first_failure_reason: str | None
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wait_for_robot_state(address: tuple[str, int], timeout: float) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        listener.bind(address)
        listener.settimeout(timeout)
        payload, _ = listener.recvfrom(65535)
        if not payload:
            raise RuntimeError("received an empty robot-state packet")
    finally:
        listener.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {path}:{line_number}"
                ) from exc
    return records


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _rate(
    records: list[dict[str, Any]], key: str, *, scale: float = 1.0
) -> float:
    if len(records) < 2:
        return 0.0
    elapsed = (float(records[-1][key]) - float(records[0][key])) * scale
    return (len(records) - 1) / elapsed if elapsed > 0.0 else 0.0


def _post_warmup_policy_records(
    records: list[dict[str, Any]], warmup_seconds: float
) -> list[dict[str, Any]]:
    if not records:
        return []
    start_ns = int(records[0]["timestamp_ns"])
    threshold_ns = start_ns + int(max(0.0, warmup_seconds) * 1e9)
    selected = [
        record
        for record in records
        if int(record["timestamp_ns"]) >= threshold_ns
    ]
    return selected or records


def _forbidden_ground_contact(record: dict[str, Any]) -> bool:
    ground_bodies = [
        str(item).lower()
        for item in record.get("ground_contact_bodies", ())
    ]
    if any(
        token in body
        for body in ground_bodies
        for token in FORBIDDEN_GROUND_BODY_TOKENS
    ):
        return True
    for contact in record.get("contacts", ()):
        geom1 = str(contact.get("geom1", "")).lower()
        geom2 = str(contact.get("geom2", "")).lower()
        if geom1 != "floor" and geom2 != "floor":
            continue
        other_body = str(
            contact.get("body2" if geom1 == "floor" else "body1", "")
        ).lower()
        if any(token in other_body for token in FORBIDDEN_GROUND_BODY_TOKENS):
            return True
    return False


def analyze_logs(
    policy_log: Path,
    simulator_log: Path,
    *,
    warmup_seconds: float = 0.5,
) -> SmokeSummary:
    policy_all = _read_jsonl(Path(policy_log))
    simulator_all = _read_jsonl(Path(simulator_log))
    policy_metadata = next(
        (
            item
            for item in policy_all
            if item.get("kind") == "run_metadata"
            and item.get("component") == "policy"
        ),
        {},
    )
    model_profile = str(
        policy_metadata.get("model", {}).get("profile", "<unknown>")
    )
    policy_records = [
        record
        for record in policy_all
        if record.get("kind") == "policy_step"
    ]
    simulator_records = [
        record
        for record in simulator_all
        if record.get("kind") == "mujoco_step"
    ]
    failures: list[str] = []
    if not policy_records:
        failures.append("policy produced no policy_step records")
    if not simulator_records:
        failures.append("MuJoCo produced no mujoco_step records")
    if failures:
        return SmokeSummary(
            model_profile=model_profile,
            policy_steps=len(policy_records),
            safe_steps=0,
            safe_fraction=0.0,
            policy_hz=0.0,
            physics_steps=len(simulator_records),
            physics_hz=0.0,
            physics_wall_hz=0.0,
            max_inference_ms=0.0,
            inference_p99_ms=0.0,
            max_projected_gravity_xy=0.0,
            max_joint_limit_violation_rad=0.0,
            forbidden_ground_contact_steps=0,
            sequence_reset_count=0,
            mid_interval_action_change_count=0,
            non_decimated_policy_step_count=0,
            first_failure_sequence=None,
            first_failure_time=None,
            first_failure_reason=None,
            passed=False,
            failures=tuple(failures),
        )
    if not all(
        _all_finite(record)
        for record in policy_records + simulator_records
    ):
        failures.append("NaN/Inf found in deployment logs")

    evaluated_policy = _post_warmup_policy_records(
        policy_records, warmup_seconds
    )
    safe_records = [
        record
        for record in evaluated_policy
        if record.get("command_armed") is True
        and record.get("command_reason") == "safe"
        and record.get("episode_failed") is not True
    ]
    safe_fraction = len(safe_records) / len(evaluated_policy)
    if len(safe_records) != len(evaluated_policy):
        first = next(
            record
            for record in evaluated_policy
            if record not in safe_records
        )
        failures.append(
            "policy was not continuously safe/armed after warm-up: "
            f"sequence={first.get('sequence')} "
            f"reason={first.get('command_reason')}"
        )

    inference = np.asarray(
        [
            float(record.get("inference_time_ms", 0.0))
            for record in evaluated_policy
        ],
        dtype=np.float64,
    )
    max_inference_ms = float(np.max(inference))
    inference_p99_ms = float(np.percentile(inference, 99.0))
    rate_policy_records = safe_records or evaluated_policy
    policy_hz = _rate(
        rate_policy_records, "timestamp_ns", scale=1e-9
    )
    physics_stride = int(
        simulator_records[-1].get("physics_step_stride", 1)
    )
    physics_hz = _rate(simulator_records, "sim_time") * physics_stride
    physics_wall_hz = (
        _rate(
            simulator_records, "wall_timestamp_ns", scale=1e-9
        )
        * physics_stride
        if all("wall_timestamp_ns" in item for item in simulator_records)
        else 0.0
    )
    if not 47.5 <= policy_hz <= 52.5:
        failures.append(
            f"policy frequency is {policy_hz:.2f} Hz, "
            "expected 50 Hz +/-5%"
        )
    if not 190.0 <= physics_hz <= 210.0:
        failures.append(
            f"physics frequency is {physics_hz:.2f} Hz, "
            "expected 200 Hz +/-5%"
        )
    if not 190.0 <= physics_wall_hz <= 210.0:
        failures.append(
            "physics wall-clock frequency is "
            f"{physics_wall_hz:.2f} Hz, expected 200 Hz +/-5%"
        )
    if inference_p99_ms >= 15.0:
        failures.append(
            f"inference p99 {inference_p99_ms:.3f} ms exceeds 15 ms"
        )

    gravity_xy = [
        float(
            record.get(
                "max_projected_gravity_xy_interval",
                np.linalg.norm(record["projected_gravity"][:2]),
            )
        )
        for record in simulator_records
        if "projected_gravity" in record
    ]
    max_gravity_xy = max(gravity_xy, default=0.0)
    if max_gravity_xy > 0.8:
        failures.append(
            "training tilt termination exceeded: "
            f"projected_gravity_xy={max_gravity_xy:.6f}"
        )

    violations = []
    for record in simulator_records:
        if "joint_limit_violation_max" in record:
            violations.append(
                float(record["joint_limit_violation_max"])
            )
        elif "joint_limit_violation" in record:
            violations.append(
                float(np.max(record["joint_limit_violation"]))
            )
    max_violation = max(violations, default=0.0)
    if max_violation >= 0.02:
        failures.append(
            f"joint hard-limit penetration {max_violation:.6f} rad >= 0.02"
        )

    forbidden_records = [
        record
        for record in simulator_records
        if _forbidden_ground_contact(record)
    ]
    forbidden_steps = len(forbidden_records)
    if forbidden_steps:
        first = forbidden_records[0]
        failures.append(
            "forbidden pelvis/torso/hip ground contact at "
            f"sim_time={float(first.get('sim_time', 0.0)):.3f}s"
        )

    simulator_failures = [
        record
        for record in simulator_records
        if record.get("episode_failed") is True
    ]
    first_failure = simulator_failures[0] if simulator_failures else None
    first_failure_sequence = (
        int(first_failure["episode_failure_sequence"])
        if first_failure is not None
        and first_failure.get("episode_failure_sequence") is not None
        else None
    )
    first_failure_time = (
        float(first_failure["episode_failure_sim_time"])
        if first_failure is not None
        and first_failure.get("episode_failure_sim_time") is not None
        else None
    )
    first_failure_reason = (
        str(first_failure.get("episode_failure_reason", "episode failed"))
        if first_failure is not None
        else None
    )
    if first_failure is None and forbidden_records:
        first_contact = forbidden_records[0]
        first_failure_sequence = int(first_contact["sequence"])
        first_failure_time = float(first_contact["sim_time"])
        ground_bodies = [
            str(item)
            for item in first_contact.get("ground_contact_bodies", ())
            if any(
                token in str(item).lower()
                for token in FORBIDDEN_GROUND_BODY_TOKENS
            )
        ]
        first_failure_reason = (
            "forbidden ground contact: "
            + ",".join(ground_bodies)
        )
    if first_failure_reason is not None:
        if first_failure is not None:
            failures.append(
                "MuJoCo episode reached a training termination at "
                f"sim_time={first_failure_time:.3f}s "
                f"sequence={first_failure_sequence}: "
                f"{first_failure_reason}"
            )

    sequence_resets = sum(
        bool(record.get("state_sequence_reset"))
        for record in policy_records
    )
    if sequence_resets:
        failures.append(
            f"unexpected robot-state sequence resets={sequence_resets}"
        )
    mid_interval_changes = 0
    active_records = [
        item
        for item in simulator_records
        if item.get("physics_started")
        and "active_command_source_sequence" in item
    ]
    if physics_stride > 1:
        for previous, current in zip(
            active_records, active_records[1:]
        ):
            if int(current["active_command_source_sequence"]) != int(
                previous["sequence"]
            ):
                mid_interval_changes += 1
    else:
        previous_source_sequence: int | None = None
        for record in active_records:
            source_sequence = int(
                record["active_command_source_sequence"]
            )
            if (
                previous_source_sequence is not None
                and source_sequence != previous_source_sequence
                and int(record.get("control_substep", -1)) != 1
            ):
                mid_interval_changes += 1
            previous_source_sequence = source_sequence
    if mid_interval_changes:
        failures.append(
            "policy target changed inside a 4-step control interval: "
            f"count={mid_interval_changes}"
        )
    policy_sequences = np.asarray(
        [int(record["sequence"]) for record in rate_policy_records],
        dtype=np.int64,
    )
    sequence_deltas = np.diff(policy_sequences)
    non_decimated_steps = int(np.count_nonzero(sequence_deltas != 4))
    if non_decimated_steps:
        first_index = int(np.flatnonzero(sequence_deltas != 4)[0])
        failures.append(
            "policy did not consume exactly every fourth physics state: "
            f"{policy_sequences[first_index]} -> "
            f"{policy_sequences[first_index + 1]}"
        )
    if model_profile == "<unknown>":
        failures.append("policy log has no validated model identity")

    return SmokeSummary(
        model_profile=model_profile,
        policy_steps=len(policy_records),
        safe_steps=len(safe_records),
        safe_fraction=safe_fraction,
        policy_hz=policy_hz,
        physics_steps=len(simulator_records) * physics_stride,
        physics_hz=physics_hz,
        physics_wall_hz=physics_wall_hz,
        max_inference_ms=max_inference_ms,
        inference_p99_ms=inference_p99_ms,
        max_projected_gravity_xy=max_gravity_xy,
        max_joint_limit_violation_rad=max_violation,
        forbidden_ground_contact_steps=forbidden_steps,
        sequence_reset_count=sequence_resets,
        mid_interval_action_change_count=mid_interval_changes,
        non_decimated_policy_step_count=non_decimated_steps,
        first_failure_sequence=first_failure_sequence,
        first_failure_time=first_failure_time,
        first_failure_reason=first_failure_reason,
        passed=not failures,
        failures=tuple(failures),
    )


def verify_logs(
    policy_log: Path,
    simulator_log: Path,
    *,
    warmup_seconds: float = 0.5,
) -> SmokeSummary:
    summary = analyze_logs(
        policy_log, simulator_log, warmup_seconds=warmup_seconds
    )
    if not summary.passed:
        raise RuntimeError("; ".join(summary.failures))
    return summary


def run_smoke(
    config_path: str | Path,
    duration: float,
    startup_timeout: float,
    log_dir: str | Path,
    *,
    profile: str | None = None,
    warmup_seconds: float = 0.5,
    raise_on_failure: bool = True,
) -> SmokeSummary:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    cfg = load_deploy_config(config_path)
    selected_profile = profile or cfg.default_policy_profile
    cfg.policy_profile(selected_profile)
    network = cfg.section("network")
    config = Path(config_path).expanduser().resolve()
    output_dir = Path(log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    simulator_log = (
        output_dir
        / f"{selected_profile}_mujoco_udp_{stamp}.jsonl"
    )
    policy_log = (
        output_dir
        / f"{selected_profile}_policy_udp_{stamp}.jsonl"
    )

    server_command = [
        sys.executable,
        "-m",
        "deploy.sim2sim.mujoco_server",
        "--config",
        str(config),
        "--transport",
        "udp",
        "--headless",
        "--log",
        str(simulator_log),
    ]
    policy_command = [
        sys.executable,
        "-m",
        "deploy.policy.run",
        "--config",
        str(config),
        "--mode",
        "sim2sim",
        "--transport",
        "udp",
        "--profile",
        selected_profile,
        "--arm",
        "--duration",
        str(duration),
        "--log",
        str(policy_log),
    ]

    server = subprocess.Popen(
        server_command,
        cwd=cfg.project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        robot_state_address = (
            str(network["robot_state_udp"][0]),
            int(network["robot_state_udp"][1]),
        )
        _wait_for_robot_state(robot_state_address, startup_timeout)
        policy = subprocess.run(
            policy_command,
            cwd=cfg.project_root,
            capture_output=True,
            text=True,
            timeout=duration + startup_timeout,
            check=False,
        )
        if policy.returncode != 0:
            raise RuntimeError(
                f"policy process exited with {policy.returncode}\n"
                f"{policy.stdout}\n{policy.stderr}"
            )
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server_output, _ = server.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server_output, _ = server.communicate(timeout=5.0)

    if server.returncode not in {0, -15, 1}:
        raise RuntimeError(
            f"MuJoCo server exited unexpectedly with {server.returncode}\n"
            f"{server_output}"
        )
    summary = analyze_logs(
        policy_log,
        simulator_log,
        warmup_seconds=warmup_seconds,
    )
    if raise_on_failure and not summary.passed:
        raise RuntimeError("; ".join(summary.failures))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--profile", help="named policy profile")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument(
        "--log-dir",
        default=str(Path.home() / "physhsi_deploy_logs"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(
        args.config,
        args.duration,
        args.startup_timeout,
        args.log_dir,
        profile=args.profile,
        warmup_seconds=args.warmup_seconds,
    )
    print("Strict UDP Sim2Sim validation passed.")
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
