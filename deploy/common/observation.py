"""Exact 123-D frame and 738-D history construction for CarryBox."""

from __future__ import annotations

import numpy as np

from .constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    DEFAULT_DOF_POS,
    FRAME_OBS_DIM,
    HISTORY_LENGTH,
    OBSERVATION_SLICES,
)
from .math_utils import (
    normalize_quat_wxyz,
    quat_rotate_inverse_wxyz,
    quat_to_tan_norm_wxyz,
)
from .types import RobotState, TaskState


class ObservationBuilder:
    """Stateful builder matching ``carrybox.py`` actor observation order."""

    def __init__(
        self,
        default_dof_pos: np.ndarray | tuple[float, ...] = DEFAULT_DOF_POS,
        *,
        ang_vel_scale: float = 0.25,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
        clip: float = 100.0,
        history_length: int = HISTORY_LENGTH,
        legacy_ankle_delay_steps: int = 1,
    ) -> None:
        if history_length != HISTORY_LENGTH:
            raise ValueError(
                f"this actor requires history_length={HISTORY_LENGTH}, got {history_length}"
            )
        if legacy_ankle_delay_steps not in (0, 1):
            raise ValueError("legacy_ankle_delay_steps currently supports only 0 or 1")
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        if self.default_dof_pos.shape != (ACTION_DIM,):
            raise ValueError("default_dof_pos must have shape (29,)")
        self.ang_vel_scale = float(ang_vel_scale)
        self.dof_pos_scale = float(dof_pos_scale)
        self.dof_vel_scale = float(dof_vel_scale)
        self.clip = float(clip)
        self.legacy_ankle_delay_steps = legacy_ankle_delay_steps
        self._history = np.zeros((HISTORY_LENGTH, FRAME_OBS_DIM), dtype=np.float32)
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._previous_endpoints: np.ndarray | None = None

    def reset(self, initial_state: RobotState | None = None) -> None:
        self._history.fill(0.0)
        self._previous_action.fill(0.0)
        self._previous_endpoints = (
            None
            if initial_state is None
            else initial_state.end_effector_pos_policy_frame.copy()
        )

    @property
    def previous_action(self) -> np.ndarray:
        return self._previous_action.copy()

    def set_previous_action(self, raw_action: np.ndarray) -> None:
        action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"raw_action must have shape ({ACTION_DIM},)")
        self._previous_action = np.clip(action, -self.clip, self.clip)

    def build_task_observation(self, task: TaskState) -> np.ndarray:
        if task.success:
            return np.full(15, -1.0, dtype=np.float64)
        relative_quat = normalize_quat_wxyz(
            task.box_quat_policy_frame_wxyz
        )
        return np.concatenate(
            (
                task.box_pos_policy_frame,
                quat_to_tan_norm_wxyz(relative_quat),
                task.box_size,
                task.goal_pos_policy_frame,
            )
        )

    def build_frame(self, robot: RobotState, task: TaskState) -> np.ndarray:
        endpoints = robot.end_effector_pos_policy_frame.copy()
        if self.legacy_ankle_delay_steps and self._previous_endpoints is not None:
            endpoints[2:4] = self._previous_endpoints[2:4]
        self._previous_endpoints = robot.end_effector_pos_policy_frame.copy()

        projected_gravity = quat_rotate_inverse_wxyz(
            robot.policy_frame_quat_wxyz, np.array([0.0, 0.0, -1.0])
        )
        frame = np.concatenate(
            (
                robot.policy_frame_ang_vel * self.ang_vel_scale,
                projected_gravity,
                (robot.joint_pos - self.default_dof_pos) * self.dof_pos_scale,
                robot.joint_vel * self.dof_vel_scale,
                endpoints.reshape(-1),
                self._previous_action,
                self.build_task_observation(task),
            )
        )
        if frame.shape != (FRAME_OBS_DIM,):
            raise AssertionError(f"frame shape is {frame.shape}, expected ({FRAME_OBS_DIM},)")
        return np.clip(frame, -self.clip, self.clip).astype(np.float32)

    def append(self, robot: RobotState, task: TaskState) -> np.ndarray:
        frame = self.build_frame(robot, task)
        self._history[:-1] = self._history[1:]
        self._history[-1] = frame
        actor_obs = self._history.reshape(1, -1)
        if actor_obs.shape != (1, ACTOR_OBS_DIM):
            raise AssertionError(
                f"actor observation shape is {actor_obs.shape}, expected (1, {ACTOR_OBS_DIM})"
            )
        return actor_obs.copy()

    def frame_slices(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        frame = np.asarray(frame)
        if frame.shape[-1] != FRAME_OBS_DIM:
            raise ValueError(f"last frame dimension must be {FRAME_OBS_DIM}")
        return {
            name: frame[..., start:end]
            for name, (start, end) in OBSERVATION_SLICES.items()
        }
