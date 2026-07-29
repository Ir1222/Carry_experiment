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
    warnings: tuple[str, ...] = ()
    latch: bool = False
    episode_failed: bool = False


class SafetyGate:
    def __init__(
        self,
        robot: RobotDescription,
        *,
        max_robot_state_age_ms: float = 40.0,
        max_task_state_age_ms: float = 100.0,
        max_projected_gravity_xy: float = 0.8,
        joint_limit_margin: float = 0.02,
        sim_joint_limit_tolerance: float = 0.02,
        profile: str = "hardware_safe",
    ) -> None:
        self.robot = robot
        self.max_robot_state_age_ns = int(max_robot_state_age_ms * 1e6)
        self.max_task_state_age_ns = int(max_task_state_age_ms * 1e6)
        self.max_projected_gravity_xy = float(max_projected_gravity_xy)
        self.joint_limit_margin = float(joint_limit_margin)
        self.sim_joint_limit_tolerance = float(sim_joint_limit_tolerance)
        self.profile = str(profile)
        if self.profile not in ("sim_parity", "hardware_safe"):
            raise ValueError(
                "safety profile must be 'sim_parity' or 'hardware_safe'"
            )
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
            return SafetyDecision(
                False, "emergency stop latched", latch=True
            )
        if not armed:
            return SafetyDecision(False, "policy not armed")
        if robot_state is None:
            return SafetyDecision(False, "no robot state", latch=True)
        if task_state is None:
            return SafetyDecision(False, "no task state", latch=True)
        if now_ns - robot_state.timestamp_ns > self.max_robot_state_age_ns:
            return SafetyDecision(False, "robot state stale", latch=True)
        if now_ns - task_state.timestamp_ns > self.max_task_state_age_ns:
            return SafetyDecision(False, "task state stale", latch=True)
        if not robot_state.is_finite():
            return SafetyDecision(
                False, "robot state is non-finite", latch=True
            )
        if not task_state.is_finite():
            return SafetyDecision(
                False, "task state is non-finite", latch=True
            )

        quat_norm = float(
            np.linalg.norm(robot_state.policy_frame_quat_wxyz)
        )
        if abs(quat_norm - 1.0) > 1e-3:
            return SafetyDecision(
                False,
                f"invalid policy-frame quaternion norm {quat_norm:.6f}",
                latch=True,
            )
        task_quat_norm = float(
            np.linalg.norm(task_state.box_quat_policy_frame_wxyz)
        )
        if abs(task_quat_norm - 1.0) > 1e-3:
            return SafetyDecision(
                False,
                f"invalid task quaternion norm {task_quat_norm:.6f}",
                latch=True,
            )
        if np.any(task_state.box_size <= 0.0):
            return SafetyDecision(
                False, "invalid non-positive box size", latch=True
            )
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_state.policy_frame_quat_wxyz,
            np.array([0.0, 0.0, -1.0]),
        )
        if np.linalg.norm(projected_gravity[:2]) > self.max_projected_gravity_xy:
            return SafetyDecision(
                False,
                "policy-frame tilt exceeds training termination threshold",
                latch=True,
                episode_failed=self.profile == "sim_parity",
            )

        warnings: list[str] = []
        if self.profile == "hardware_safe":
            lower = self.robot.lower_limits + self.joint_limit_margin
            upper = self.robot.upper_limits - self.joint_limit_margin
            invalid = np.flatnonzero(
                (robot_state.joint_pos < lower)
                | (robot_state.joint_pos > upper)
            )
            if invalid.size:
                names = ",".join(
                    self.robot.joint_names[int(index)]
                    for index in invalid[:4]
                )
                return SafetyDecision(
                    False,
                    f"joint position outside hardware-safe limits: {names}",
                    latch=True,
                )
        else:
            below = self.robot.lower_limits - robot_state.joint_pos
            above = robot_state.joint_pos - self.robot.upper_limits
            violation = np.maximum(np.maximum(below, above), 0.0)
            worst_index = int(np.argmax(violation))
            worst = float(violation[worst_index])
            if worst > self.sim_joint_limit_tolerance:
                return SafetyDecision(
                    False,
                    "sim joint-limit penetration "
                    f"{self.robot.joint_names[worst_index]}={worst:.6f} rad "
                    f"> {self.sim_joint_limit_tolerance:.6f} rad",
                    latch=True,
                    episode_failed=True,
                )
            near = np.flatnonzero(
                (robot_state.joint_pos < self.robot.lower_limits + self.joint_limit_margin)
                | (robot_state.joint_pos > self.robot.upper_limits - self.joint_limit_margin)
            )
            if near.size:
                names = ",".join(
                    self.robot.joint_names[int(index)]
                    for index in near[:4]
                )
                warnings.append(f"joint near/through hard limit: {names}")
        if dry_run:
            return SafetyDecision(
                False,
                "dry-run blocks command output",
                warnings=tuple(warnings),
            )
        return SafetyDecision(True, "safe", warnings=tuple(warnings))
