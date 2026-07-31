"""Verify PyTorch/ONNX first-step actor parity on exact Isaac snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from deploy.common.config import load_deploy_config
from deploy.common.snapshot import CarryBoxSnapshot
from deploy.policy.inference import OnnxActor
from deploy.tools.export_actor import load_actor_from_checkpoint


def _profile_names(cfg, value: str | None) -> tuple[str, ...]:
    configured = tuple(cfg.section("policy").get("profiles", {}).keys())
    if value is None:
        return configured
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(configured))
    if not selected or unknown:
        raise ValueError(f"invalid profiles; unknown={unknown}, available={configured}")
    return selected


def compare_actor_snapshot(
    snapshot_path: str | Path,
    config_path: str | Path,
    *,
    profiles: str | None = None,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    cfg = load_deploy_config(config_path)
    snapshot = CarryBoxSnapshot.load(snapshot_path)
    actor_obs = snapshot.actor_obs.reshape(1, -1)
    results: dict[str, object] = {}
    failures: list[str] = []
    for profile in _profile_names(cfg, profiles):
        actor = load_actor_from_checkpoint(cfg.checkpoint_path_for(profile))
        with torch.inference_mode():
            torch_action = actor(torch.from_numpy(actor_obs)).numpy()[0]
        onnx_action = OnnxActor(cfg.onnx_path_for(profile))(actor_obs)[0]
        backend_error = float(np.max(np.abs(torch_action - onnx_action)))
        stored_error = (
            None
            if snapshot.policy_action is None
            else float(np.max(np.abs(torch_action - snapshot.policy_action)))
        )
        results[profile] = {
            "torch_onnx_max_abs_error": backend_error,
            "torch_action": torch_action.tolist(),
            "onnx_action": onnx_action.tolist(),
            "stored_policy_action_max_abs_error": stored_error,
        }
        if backend_error >= tolerance:
            failures.append(
                f"{profile}: PyTorch/ONNX error {backend_error:.8g} >= {tolerance}"
            )
    report = {
        "snapshot": str(snapshot.path),
        "phase": snapshot.phase,
        "tolerance": float(tolerance),
        "profiles": results,
    }
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument(
        "--profiles",
        help="comma-separated profiles; default is every configured profile",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    report = compare_actor_snapshot(
        args.snapshot,
        args.config,
        profiles=args.profiles,
        tolerance=args.tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
