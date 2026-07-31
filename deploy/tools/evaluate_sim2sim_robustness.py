"""Evaluate CarryBox policies on paired seeded MuJoCo scenarios."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from deploy.common.config import load_deploy_config
from deploy.sim2sim.collision_profiles import COLLISION_PROFILE_NAMES
from deploy.sim2sim.randomization import PROFILE_NAMES, ScenarioSampler
from deploy.tools.run_udp_smoke import run_smoke


def _model_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "--models must contain unique comma-separated profile names"
        )
    return result


def _seed_values(value: str) -> tuple[int, ...]:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "--seeds range must be START:STOP[:STEP]"
            )
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise argparse.ArgumentTypeError("--seeds step cannot be zero")
        result = tuple(range(start, stop, step))
    else:
        result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--seeds must contain unique values")
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    return (
        None
        if not values
        else float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
    )


def _aggregate(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model_profile"] == model]
    infrastructure = [
        row for row in selected if row["outcome"] == "infrastructure_error"
    ]
    completed = [
        row for row in selected if row["outcome"] != "infrastructure_error"
    ]
    successes = [row for row in completed if row["outcome"] == "success"]
    grasp_successes = [
        row for row in completed if bool(row.get("grasp_success", False))
    ]
    terminations = [row for row in completed if row["outcome"] == "termination"]
    timeouts = [row for row in completed if row["outcome"] == "timeout"]
    failure_times = [
        float(row["first_failure_time"])
        for row in terminations
        if row.get("first_failure_time") is not None
    ]
    reasons = Counter(
        str(row.get("first_failure_code") or "unknown")
        for row in terminations
    )
    denominator = len(completed)
    return {
        "model_profile": model,
        "episode_count": len(selected),
        "completed_episode_count": denominator,
        "infrastructure_error_count": len(infrastructure),
        "success_count": len(successes),
        "termination_count": len(terminations),
        "timeout_count": len(timeouts),
        "success_rate": len(successes) / denominator if denominator else 0.0,
        "grasp_success_count": len(grasp_successes),
        "grasp_success_rate": (
            len(grasp_successes) / denominator if denominator else 0.0
        ),
        "termination_rate": (
            len(terminations) / denominator if denominator else 0.0
        ),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "failure_time_median": _percentile(failure_times, 50.0),
        "failure_time_p10": _percentile(failure_times, 10.0),
        "max_tilt_mean": (
            float(np.mean([row["max_projected_gravity_xy"] for row in completed]))
            if completed
            else None
        ),
        "peak_abs_torque_p95": _percentile(
            [float(row["peak_abs_torque"]) for row in completed], 95.0
        ),
        "max_contact_force_p95": _percentile(
            [float(row["max_contact_force"]) for row in completed], 95.0
        ),
        "min_left_palm_box_signed_distance_p10": _percentile(
            [
                float(row["min_left_palm_box_signed_distance"])
                for row in completed
                if row.get("min_left_palm_box_signed_distance") is not None
            ],
            10.0,
        ),
        "min_right_palm_box_signed_distance_p10": _percentile(
            [
                float(row["min_right_palm_box_signed_distance"])
                for row in completed
                if row.get("min_right_palm_box_signed_distance") is not None
            ],
            10.0,
        ),
        "anomalous_self_contact_steps_total": sum(
            int(row.get("anomalous_self_contact_steps", 0))
            for row in completed
        ),
        "initial_anomalous_self_contact_count": sum(
            bool(row.get("initial_anomalous_self_contact", False))
            for row in completed
        ),
        "hip_or_torso_box_contact_steps_total": sum(
            int(row.get("hip_or_torso_box_contact_steps", 0))
            for row in completed
        ),
        "distinct_trace_hash_count": len(
            {
                row["trace_hash"]
                for row in completed
                if row.get("trace_hash") is not None
            }
        ),
    }


def evaluate_robustness(
    config_path: str | Path,
    models: Sequence[str],
    seeds: Sequence[int],
    *,
    randomization_profile: str,
    duration: float,
    startup_timeout: float,
    warmup_seconds: float,
    report_dir: str | Path,
    collision_profile: str = "current",
    goal_profile: str = "configured",
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    if randomization_profile == "nominal":
        raise ValueError(
            "robustness evaluation requires light or train_match; use "
            "validate_sim2sim for nominal regression"
        )
    cfg = load_deploy_config(config_path)
    for model in models:
        cfg.policy_profile(model)
    sampler = ScenarioSampler(cfg)
    root = (
        Path(report_dir).expanduser().resolve()
        / datetime.now().strftime("robustness_%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir()
    model_dirs: dict[str, Path] = {}
    for model in models:
        model_dirs[model] = root / model
        model_dirs[model].mkdir()

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        scenario = sampler.sample(
            randomization_profile, int(seed), episode_index=0
        )
        scenario_path = scenario.write(
            scenario_dir / f"seed_{int(seed)}.json"
        )
        for model in models:
            run_dir = model_dirs[model] / f"seed_{int(seed)}"
            run_dir.mkdir()
            print(
                f"[RUN] model={model} profile={randomization_profile} "
                f"collision={collision_profile} seed={seed} "
                f"duration={duration:g}s"
            )
            try:
                summary = run_smoke(
                    config_path,
                    duration,
                    startup_timeout,
                    run_dir,
                    profile=model,
                    warmup_seconds=warmup_seconds,
                    raise_on_failure=False,
                    randomization_profile=randomization_profile,
                    seed=int(seed),
                    scenario_file=scenario_path,
                    collision_profile=collision_profile,
                    goal_profile=goal_profile,
                )
                row = summary.to_dict()
            except Exception as exc:
                row = {
                    "model_profile": model,
                    "randomization_profile": randomization_profile,
                    "seed": int(seed),
                    "scenario_fingerprint": scenario.fingerprint,
                    "outcome": "infrastructure_error",
                    "ever_success": False,
                    "time_to_success": None,
                    "first_failure_time": None,
                    "first_failure_code": None,
                    "first_failure_reason": None,
                    "max_projected_gravity_xy": 0.0,
                    "peak_abs_torque": 0.0,
                    "max_contact_force": 0.0,
                    "collision_profile": collision_profile,
                    "grasp_success": False,
                    "trace_hash": None,
                    "passed": False,
                    "failures": [
                        f"{type(exc).__name__}: {exc}"
                    ],
                }
            row["scenario_path"] = str(scenario_path)
            row["run_dir"] = str(run_dir)
            rows.append(row)
            print(
                f"[{row['outcome'].upper()}] model={model} seed={seed} "
                f"trace={str(row.get('trace_hash'))[:12]}"
            )

    aggregate = [_aggregate(rows, model) for model in models]
    paired = []
    for seed in seeds:
        by_model = {
            row["model_profile"]: {
                "outcome": row["outcome"],
                "first_failure_time": row.get("first_failure_time"),
                "trace_hash": row.get("trace_hash"),
                "grasp_success": bool(row.get("grasp_success", False)),
            }
            for row in rows
            if int(row["seed"]) == int(seed)
        }
        paired.append({"seed": int(seed), "models": by_model})
    report = {
        "format_version": 1,
        "config": str(Path(config_path).expanduser().resolve()),
        "randomization_profile": randomization_profile,
        "collision_profile": collision_profile,
        "goal_profile": goal_profile,
        "seeds": [int(seed) for seed in seeds],
        "duration_seconds": float(duration),
        "warmup_seconds": float(warmup_seconds),
        "models": aggregate,
        "paired_results": paired,
        "episodes": rows,
    }
    report_path = root / "robustness_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = root / "robustness_episodes.csv"
    csv_columns = (
        "model_profile",
        "randomization_profile",
        "seed",
        "scenario_fingerprint",
        "outcome",
        "ever_success",
        "time_to_success",
        "first_failure_time",
        "first_failure_code",
        "first_failure_reason",
        "max_projected_gravity_xy",
        "max_joint_limit_violation_rad",
        "forbidden_ground_contact_steps",
        "peak_abs_torque",
        "max_contact_force",
        "collision_profile",
        "grasp_success",
        "first_left_hand_contact_time",
        "first_right_hand_contact_time",
        "first_bimanual_contact_time",
        "max_bimanual_contact_duration",
        "max_box_clearance",
        "min_left_palm_box_signed_distance",
        "min_right_palm_box_signed_distance",
        "anomalous_self_contact_steps",
        "initial_anomalous_self_contact",
        "hip_or_torso_box_contact_steps",
        "trace_hash",
        "passed",
        "run_dir",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report_path, rows, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument(
        "--models",
        type=_model_names,
        default=_model_names(
            "official_carrybox_65000,model_55500,model_73500"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_seed_values,
        default=_seed_values("0:10"),
        help="comma-separated seeds or START:STOP[:STEP]",
    )
    parser.add_argument(
        "--randomization-profile",
        choices=PROFILE_NAMES,
        default="light",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument(
        "--collision-profile",
        choices=COLLISION_PROFILE_NAMES,
        default="current",
    )
    parser.add_argument(
        "--goal-profile",
        choices=("configured", "isaac_sector"),
        default="configured",
    )
    parser.add_argument(
        "--report-dir",
        default=str(Path.home() / "physhsi_deploy_logs"),
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        help="optional CI gate applied independently to every model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_success_rate is not None and not (
        0.0 <= args.min_success_rate <= 1.0
    ):
        raise ValueError("--min-success-rate must be in [0, 1]")
    report_path, rows, aggregate = evaluate_robustness(
        args.config,
        args.models,
        args.seeds,
        randomization_profile=args.randomization_profile,
        duration=args.duration,
        startup_timeout=args.startup_timeout,
        warmup_seconds=args.warmup_seconds,
        report_dir=args.report_dir,
        collision_profile=args.collision_profile,
        goal_profile=args.goal_profile,
    )
    print(f"Robustness report: {report_path}")
    if any(row["outcome"] == "infrastructure_error" for row in rows):
        return 2
    if args.min_success_rate is not None and any(
        float(item["success_rate"]) < args.min_success_rate for item in aggregate
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
