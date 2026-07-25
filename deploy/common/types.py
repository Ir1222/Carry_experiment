"""Typed state boundaries shared by simulation and hardware backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import ACTION_DIM


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    return result


@dataclass(slots=True)
class RobotState:
    """Policy-ready robot state.

    Quaternion order is WXYZ. End-effector positions are already expressed in
    the torso frame after subtracting the pelvis/root translation, matching the
    training environment.
    """

    sequence: int
    timestamp_ns: int
    torso_quat_wxyz: np.ndarray
    torso_ang_vel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    end_effector_pos_torso: np.ndarray

    def __post_init__(self) -> None:
        self.torso_quat_wxyz = _vector(self.torso_quat_wxyz, 4, "torso_quat_wxyz")
        self.torso_ang_vel = _vector(self.torso_ang_vel, 3, "torso_ang_vel")
        self.joint_pos = _vector(self.joint_pos, ACTION_DIM, "joint_pos")
        self.joint_vel = _vector(self.joint_vel, ACTION_DIM, "joint_vel")
        endpoints = np.asarray(self.end_effector_pos_torso, dtype=np.float64)
        if endpoints.size != 15:
            raise ValueError(
                "end_effector_pos_torso must contain 5 xyz positions (15 values), "
                f"got shape {endpoints.shape}"
            )
        self.end_effector_pos_torso = endpoints.reshape(5, 3)

    def is_finite(self) -> bool:
        arrays = (
            self.torso_quat_wxyz,
            self.torso_ang_vel,
            self.joint_pos,
            self.joint_vel,
            self.end_effector_pos_torso,
        )
        return all(np.isfinite(item).all() for item in arrays)


@dataclass(slots=True)
class TaskState:
    """Torso-relative box and goal state produced by simulation or perception."""

    sequence: int
    timestamp_ns: int
    box_pos_torso: np.ndarray
    box_quat_torso_wxyz: np.ndarray
    box_size: np.ndarray
    goal_pos_torso: np.ndarray
    success: bool = False

    def __post_init__(self) -> None:
        self.box_pos_torso = _vector(self.box_pos_torso, 3, "box_pos_torso")
        self.box_quat_torso_wxyz = _vector(
            self.box_quat_torso_wxyz, 4, "box_quat_torso_wxyz"
        )
        self.box_size = _vector(self.box_size, 3, "box_size")
        self.goal_pos_torso = _vector(self.goal_pos_torso, 3, "goal_pos_torso")

    def is_finite(self) -> bool:
        arrays = (
            self.box_pos_torso,
            self.box_quat_torso_wxyz,
            self.box_size,
            self.goal_pos_torso,
        )
        return all(np.isfinite(item).all() for item in arrays)


@dataclass(slots=True)
class PolicyCommand:
    """Raw policy output and the low-level position command derived from it."""

    sequence: int
    timestamp_ns: int
    raw_action: np.ndarray
    q_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM))
    armed: bool = False
    reason: str = "not armed"

    def __post_init__(self) -> None:
        self.raw_action = _vector(self.raw_action, ACTION_DIM, "raw_action")
        self.q_target = _vector(self.q_target, ACTION_DIM, "q_target")
        self.kp = _vector(self.kp, ACTION_DIM, "kp")
        self.kd = _vector(self.kd, ACTION_DIM, "kd")
        self.tau_ff = _vector(self.tau_ff, ACTION_DIM, "tau_ff")

    def is_finite(self) -> bool:
        return all(
            np.isfinite(item).all()
            for item in (
                self.raw_action,
                self.q_target,
                self.kp,
                self.kd,
                self.tau_ff,
            )
        )
