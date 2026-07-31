"""Compare deployment observation, MuJoCo FK, and grasp geometry to Isaac snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from deploy.common.config import load_deploy_config
from deploy.common.grasp_diagnostics import point_to_obb_signed_distance
from deploy.common.kinematics import MujocoKinematicsProvider
from deploy.common.mapping import RobotDescription
from deploy.common.snapshot import CarryBoxSnapshot
from deploy.tools.compare_observation_snapshot import compare_snapshot


def compare_carrybox_parity(
    snapshot_paths: Sequence[str | Path],
    config_path: str | Path,
    *,
    fk_tolerance_m: float = 1e-3,
    observation_tolerance: float = 1e-5,
) -> dict[str, object]:
    cfg = load_deploy_config(config_path)
    robot = RobotDescription.from_urdf(cfg.urdf_path)
    provider = MujocoKinematicsProvider(
        cfg.resolve_path(cfg.section("simulation")["generated_robot_mjcf"]),
        robot,
        pelvis_body=cfg.section("robot")["base_body"],
        torso_body=cfg.section("robot")["torso_body"],
        policy_frame_body=cfg.section("robot")["policy_frame"],
        end_effector_names=tuple(cfg.section("robot")["end_effectors"]),
    )
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for path in snapshot_paths:
        snapshot = CarryBoxSnapshot.load(path)
        observation = compare_snapshot(
            snapshot.path, tolerance=observation_tolerance
        )
        mujoco_endpoints = provider.endpoints(snapshot.joint_pos)
        endpoint_delta = mujoco_endpoints - snapshot.end_effector_pos_policy_frame
        endpoint_errors = np.linalg.norm(endpoint_delta, axis=1)
        max_fk_error = float(np.max(endpoint_errors))
        if max_fk_error > fk_tolerance_m:
            failures.append(
                f"{snapshot.path.name}: FK error {max_fk_error:.8g} m "
                f"> {fk_tolerance_m} m"
            )
        geometry: dict[str, float] = {}
        with np.load(snapshot.path, allow_pickle=False) as data:
            if "hand_collision_position_world" in data.files:
                hand_positions = np.asarray(
                    data["hand_collision_position_world"], dtype=np.float64
                ).reshape(2, 3)
                geometry = {
                    "left_rubber_hand_box_signed_distance": (
                        point_to_obb_signed_distance(
                            hand_positions[0],
                            snapshot.box_position_world,
                            snapshot.box_quaternion_wxyz,
                            snapshot.box_size,
                        )
                    ),
                    "right_rubber_hand_box_signed_distance": (
                        point_to_obb_signed_distance(
                            hand_positions[1],
                            snapshot.box_position_world,
                            snapshot.box_quaternion_wxyz,
                            snapshot.box_size,
                        )
                    ),
                }
            if "end_effector_position_world" in data.files:
                palms = np.asarray(
                    data["end_effector_position_world"], dtype=np.float64
                ).reshape(5, 3)[:2]
                geometry.update(
                    {
                        "left_palm_box_signed_distance": point_to_obb_signed_distance(
                            palms[0],
                            snapshot.box_position_world,
                            snapshot.box_quaternion_wxyz,
                            snapshot.box_size,
                        ),
                        "right_palm_box_signed_distance": point_to_obb_signed_distance(
                            palms[1],
                            snapshot.box_position_world,
                            snapshot.box_quaternion_wxyz,
                            snapshot.box_size,
                        ),
                    }
                )
        rows.append(
            {
                "snapshot": str(snapshot.path),
                "phase": snapshot.phase,
                "observation": observation,
                "mujoco_endpoint_positions_policy_frame": mujoco_endpoints.tolist(),
                "isaac_endpoint_positions_policy_frame": (
                    snapshot.end_effector_pos_policy_frame.tolist()
                ),
                "endpoint_error_vectors_m": endpoint_delta.tolist(),
                "endpoint_error_norms_m": endpoint_errors.tolist(),
                "max_fk_error_m": max_fk_error,
                "geometry": geometry,
            }
        )
    report = {
        "fk_tolerance_m": float(fk_tolerance_m),
        "observation_tolerance": float(observation_tolerance),
        "snapshots": rows,
    }
    if failures:
        raise RuntimeError("; ".join(failures) + "\n" + json.dumps(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+")
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--fk-tolerance-m", type=float, default=1e-3)
    parser.add_argument("--observation-tolerance", type=float, default=1e-5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = compare_carrybox_parity(
        args.snapshots,
        args.config,
        fk_tolerance_m=args.fk_tolerance_m,
        observation_tolerance=args.observation_tolerance,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(
            text + "\n", encoding="utf-8"
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
