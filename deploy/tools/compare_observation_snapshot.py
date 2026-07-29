"""Compare an Isaac Gym CarryBox snapshot with deployment observation math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from deploy.common.constants import FRAME_OBS_DIM
from deploy.common.math_utils import (
    quat_relative_wxyz,
    quat_rotate_inverse_wxyz,
    xyzw_to_wxyz,
)
from deploy.common.observation import ObservationBuilder
from deploy.common.types import RobotState, TaskState


def compare_snapshot(
    path: str | Path, *, tolerance: float = 1e-5
) -> dict[str, float]:
    snapshot_path = Path(path).expanduser().resolve()
    with np.load(snapshot_path, allow_pickle=False) as data:
        values = {name: np.asarray(data[name]) for name in data.files}

    policy_quat = xyzw_to_wxyz(values["policy_frame_quat_xyzw"])
    box_quat = xyzw_to_wxyz(values["box_quat_xyzw"])
    root_position = values["root_position_world"]
    box_position = values["box_position_world"]
    goal_position = values["goal_position_world"]
    robot = RobotState(
        sequence=1,
        timestamp_ns=1,
        policy_frame_quat_wxyz=policy_quat,
        policy_frame_ang_vel=values["policy_frame_ang_vel"],
        joint_pos=values["joint_pos"],
        joint_vel=values["joint_vel"],
        end_effector_pos_policy_frame=values[
            "end_effector_pos_policy_frame"
        ],
    )
    task = TaskState(
        sequence=1,
        timestamp_ns=1,
        box_pos_policy_frame=quat_rotate_inverse_wxyz(
            policy_quat, box_position - root_position
        ),
        box_quat_policy_frame_wxyz=quat_relative_wxyz(
            policy_quat, box_quat
        ),
        box_size=values["box_size"],
        goal_pos_policy_frame=quat_rotate_inverse_wxyz(
            policy_quat, goal_position - root_position
        ),
        success=bool(values["success"]),
    )
    builder = ObservationBuilder(legacy_ankle_delay_steps=0)
    builder.set_previous_action(values["previous_action"])
    deployed_frame = builder.build_frame(robot, task)
    isaac_frame = values["current_frame"].astype(np.float32)
    actor_obs = values["actor_obs"].astype(np.float32)
    if isaac_frame.shape != (FRAME_OBS_DIM,):
        raise ValueError(
            f"snapshot current_frame has shape {isaac_frame.shape}"
        )
    if actor_obs.shape != (738,):
        raise ValueError(f"snapshot actor_obs has shape {actor_obs.shape}")

    frame_error = float(np.max(np.abs(deployed_frame - isaac_frame)))
    history_tail_error = float(
        np.max(np.abs(actor_obs[-FRAME_OBS_DIM:] - isaac_frame))
    )
    torso_policy_quat_difference = float(
        np.max(
            np.abs(
                xyzw_to_wxyz(values["torso_quat_xyzw"]) - policy_quat
            )
        )
    )
    result = {
        "frame_max_abs_error": frame_error,
        "history_tail_max_abs_error": history_tail_error,
        "torso_policy_quat_difference": torso_policy_quat_difference,
    }
    if frame_error >= tolerance or history_tail_error >= tolerance:
        raise RuntimeError(
            "observation parity failed: " + json.dumps(result, sort_keys=True)
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    result = compare_snapshot(args.snapshot, tolerance=args.tolerance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
