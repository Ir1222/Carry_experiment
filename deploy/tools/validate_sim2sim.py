"""Validate one or more named CarryBox actors in UDP MuJoCo Sim2Sim."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

from deploy.common.config import load_deploy_config
from deploy.tools.run_udp_smoke import run_smoke


def _model_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("--models must contain at least one profile")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("--models contains duplicate profiles")
    return names


def validate_models(
    config_path: str | Path,
    models: Sequence[str],
    *,
    duration: float,
    startup_timeout: float,
    warmup_seconds: float,
    report_dir: str | Path,
) -> tuple[Path, list[dict]]:
    cfg = load_deploy_config(config_path)
    for profile in models:
        cfg.policy_profile(profile)
    root = (
        Path(report_dir).expanduser().resolve()
        / datetime.now().strftime("sim2sim_%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)
    results: list[dict] = []
    for profile in models:
        model_dir = root / profile
        model_dir.mkdir()
        print(f"[RUN] profile={profile} duration={duration:g}s")
        try:
            summary = run_smoke(
                config_path,
                duration,
                startup_timeout,
                model_dir,
                profile=profile,
                warmup_seconds=warmup_seconds,
                raise_on_failure=False,
            )
            result = summary.to_dict()
        except Exception as exc:
            result = {
                "model_profile": profile,
                "passed": False,
                "failures": [
                    f"process/infrastructure failure: "
                    f"{type(exc).__name__}: {exc}"
                ],
            }
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] profile={profile}")
        for failure in result.get("failures", ()):
            print(f"  - {failure}")

    report = {
        "format_version": 1,
        "config": str(Path(config_path).resolve()),
        "duration_seconds": float(duration),
        "warmup_seconds": float(warmup_seconds),
        "all_passed": all(bool(item["passed"]) for item in results),
        "models": results,
    }
    report_path = root / "validation_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument(
        "--models",
        type=_model_names,
        default=_model_names(
            "official_carrybox_65000,model_73500"
        ),
        help="comma-separated named policy profiles",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument(
        "--report-dir",
        default=str(Path.home() / "physhsi_deploy_logs"),
    )
    # Kept for command-line compatibility; validation is always headless.
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path, results = validate_models(
        args.config,
        args.models,
        duration=args.duration,
        startup_timeout=args.startup_timeout,
        warmup_seconds=args.warmup_seconds,
        report_dir=args.report_dir,
    )
    passed = all(bool(item["passed"]) for item in results)
    print(f"Validation report: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
