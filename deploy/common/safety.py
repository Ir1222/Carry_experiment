"""Fail-closed safety gate used by simulation and hardware runners."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .mapping import RobotDescription
from .math_utils import quat_rotate_inverse_wxyz
from .types import RobotState, TaskState


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    reason: str


class SafetyGate:
    def __init__(
        self,
        robot: RobotDescription,
        *,
        max_robot_state_age_ms: float = 40.0,
        max_task_state_age_ms: float = 100.0,
        max_projected_gravity_xy: float = 0.8,
        joint_limit_margin: float = 0.02,
    ) -> None:
        self.robot = robot
        self.max_robot_state_age_ns = int(max_robot_state_age_ms * 1e6)
        self.max_task_state_age_ns = int(max_task_state_age_ms * 1e6)
        self.max_projected_gravity_xy = float(max_projected_gravity_xy)
        self.joint_limit_margin = float(joint_limit_margin)
        self.estop_latched = False

    def trigger_estop(self) -> None:
        self.estop_latched = True

    def clear_estop(self) -> None:
        self.estop_latched = False

    def evaluate(
        self,
        robot_state: RobotState | None,
        task_state: TaskState | None,
        *,
        armed: bool,
        dry_run: bool,
        now_ns: int | None = None,
    ) -> SafetyDecision:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        if self.estop_latched:
            return SafetyDecision(False, "emergency stop latched")
        if not armed:
            return SafetyDecision(False, "policy not armed")
        if robot_state is None:
            return SafetyDecision(False, "no robot state")
        if task_state is None:
            return SafetyDecision(False, "no task state")
        if now_ns - robot_state.timestamp_ns > self.max_robot_state_age_ns:
            return SafetyDecision(False, "robot state stale")
        if now_ns - task_state.timestamp_ns > self.max_task_state_age_ns:
            return SafetyDecision(False, "task state stale")
        if not robot_state.is_finite():
            return SafetyDecision(False, "robot state is non-finite")
        if not task_state.is_finite():
            return SafetyDecision(False, "task state is non-finite")

        quat_norm = float(np.linalg.norm(robot_state.torso_quat_wxyz))
        if abs(quat_norm - 1.0) > 1e-3:
            return SafetyDecision(False, f"invalid torso quaternion norm {quat_norm:.6f}")
        task_quat_norm = float(np.linalg.norm(task_state.box_quat_torso_wxyz))
        if abs(task_quat_norm - 1.0) > 1e-3:
            return SafetyDecision(
                False, f"invalid task quaternion norm {task_quat_norm:.6f}"
            )
        if np.any(task_state.box_size <= 0.0):
            return SafetyDecision(False, "invalid non-positive box size")
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_state.torso_quat_wxyz, np.array([0.0, 0.0, -1.0])
        )
        if np.linalg.norm(projected_gravity[:2]) > self.max_projected_gravity_xy:
            return SafetyDecision(False, "torso tilt exceeds safety threshold")

        lower = self.robot.lower_limits - self.joint_limit_margin
        upper = self.robot.upper_limits + self.joint_limit_margin
        if np.any(robot_state.joint_pos < lower) or np.any(
            robot_state.joint_pos > upper
        ):
            return SafetyDecision(False, "joint position outside validated limits")
        if dry_run:
            return SafetyDecision(False, "dry-run blocks command output")
        return SafetyDecision(True, "safe")
