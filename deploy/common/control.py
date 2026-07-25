"""Policy action scaling and PhysHSI PD control."""

from __future__ import annotations

import time

import numpy as np

from .constants import ACTION_DIM, DEFAULT_DOF_POS, KD, KP
from .mapping import RobotDescription
from .types import PolicyCommand


class PDController:
    def __init__(
        self,
        robot: RobotDescription,
        *,
        default_dof_pos: np.ndarray | tuple[float, ...] = DEFAULT_DOF_POS,
        kp: np.ndarray | tuple[float, ...] = KP,
        kd: np.ndarray | tuple[float, ...] = KD,
        action_scale: float = 0.25,
        action_clip: float = 100.0,
    ) -> None:
        self.robot = robot
        self.default_dof_pos = self._array(default_dof_pos, "default_dof_pos")
        self.kp = self._array(kp, "kp")
        self.kd = self._array(kd, "kd")
        self.action_scale = float(action_scale)
        self.action_clip = float(action_clip)

    @staticmethod
    def _array(value: np.ndarray | tuple[float, ...], name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (ACTION_DIM,):
            raise ValueError(f"{name} must have shape ({ACTION_DIM},)")
        return result

    def action_to_target(
        self, raw_action: np.ndarray, *, clamp_joint_limits: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"raw_action must have shape ({ACTION_DIM},)")
        clipped_action = np.clip(action, -self.action_clip, self.action_clip)
        q_target = self.default_dof_pos + self.action_scale * clipped_action
        if clamp_joint_limits:
            q_target = np.clip(
                q_target, self.robot.lower_limits, self.robot.upper_limits
            )
        return clipped_action, q_target

    def compute_torque(
        self,
        q_target: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        tau_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        q_target = self._array(q_target, "q_target")
        joint_pos = self._array(joint_pos, "joint_pos")
        joint_vel = self._array(joint_vel, "joint_vel")
        feedforward = (
            np.zeros(ACTION_DIM, dtype=np.float64)
            if tau_ff is None
            else self._array(tau_ff, "tau_ff")
        )
        torque = (
            feedforward
            + self.kp * float(kp_scale) * (q_target - joint_pos)
            - self.kd * float(kd_scale) * joint_vel
        )
        return np.clip(
            torque, -self.robot.effort_limits, self.robot.effort_limits
        )

    def policy_command(
        self,
        raw_action: np.ndarray,
        *,
        sequence: int,
        armed: bool,
        reason: str,
        hardware_safe: bool,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> PolicyCommand:
        clipped_action, q_target = self.action_to_target(
            raw_action, clamp_joint_limits=hardware_safe
        )
        return PolicyCommand(
            sequence=sequence,
            timestamp_ns=time.monotonic_ns(),
            raw_action=clipped_action,
            q_target=q_target,
            kp=self.kp * float(kp_scale),
            kd=self.kd * float(kd_scale),
            armed=armed,
            reason=reason,
        )

    def hold_command(
        self,
        joint_pos: np.ndarray,
        *,
        sequence: int,
        reason: str,
        damping_scale: float = 1.0,
        armed: bool = False,
    ) -> PolicyCommand:
        joint_pos = self._array(joint_pos, "joint_pos")
        return PolicyCommand(
            sequence=sequence,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(ACTION_DIM),
            q_target=joint_pos.copy(),
            kp=np.zeros(ACTION_DIM),
            kd=self.kd * float(damping_scale),
            armed=armed,
            reason=reason,
        )
