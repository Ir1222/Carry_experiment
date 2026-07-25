"""Backend-independent observation → actor → command pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from deploy.common.constants import ACTION_DIM, ACTOR_OBS_DIM
from deploy.common.control import PDController
from deploy.common.observation import ObservationBuilder
from deploy.common.types import PolicyCommand, RobotState, TaskState


@dataclass(slots=True)
class PolicyStep:
    actor_obs: np.ndarray
    raw_action: np.ndarray
    command: PolicyCommand
    inference_time_ms: float


class PolicyCore:
    def __init__(
        self,
        actor,
        observation_builder: ObservationBuilder,
        controller: PDController,
        *,
        max_inference_time_ms: float = 15.0,
    ) -> None:
        self.actor = actor
        self.observation_builder = observation_builder
        self.controller = controller
        self.max_inference_time_ms = float(max_inference_time_ms)
        self.sequence = 0
        self._last_finite_joint_pos = self.controller.default_dof_pos.copy()

    def reset(self, initial_state: RobotState | None = None) -> None:
        self.observation_builder.reset(initial_state)
        self.sequence = 0
        if initial_state is not None and np.isfinite(initial_state.joint_pos).all():
            self._last_finite_joint_pos = initial_state.joint_pos.copy()

    def step(
        self,
        robot_state: RobotState,
        task_state: TaskState,
        *,
        command_allowed: bool,
        reason: str,
        hardware_safe: bool,
        update_action_history: bool = True,
        run_inference_when_blocked: bool = False,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> PolicyStep:
        if np.isfinite(robot_state.joint_pos).all():
            self._last_finite_joint_pos = robot_state.joint_pos.copy()
        if not command_allowed and not run_inference_when_blocked:
            self.sequence += 1
            return PolicyStep(
                actor_obs=np.zeros((1, ACTOR_OBS_DIM), dtype=np.float32),
                raw_action=np.zeros(ACTION_DIM, dtype=np.float64),
                command=self.controller.hold_command(
                    self._last_finite_joint_pos,
                    sequence=self.sequence,
                    reason=reason,
                    armed=False,
                ),
                inference_time_ms=0.0,
            )
        actor_obs = self.observation_builder.append(robot_state, task_state)
        inference_start = time.perf_counter_ns()
        inference_error: Exception | None = None
        try:
            actor_output = np.asarray(self.actor(actor_obs), dtype=np.float64)
            if actor_output.shape != (1, ACTION_DIM):
                raise ValueError(
                    f"actor output must have shape (1, {ACTION_DIM}), "
                    f"got {actor_output.shape}"
                )
        except Exception as exc:  # fail closed at the policy/hardware boundary
            inference_error = exc
            actor_output = np.zeros((1, ACTION_DIM), dtype=np.float64)
        inference_time_ms = (time.perf_counter_ns() - inference_start) / 1e6
        raw_action = actor_output[0]
        action_valid = bool(np.isfinite(raw_action).all())
        if inference_error is not None:
            command_allowed = False
            reason = (
                "actor inference failed: "
                f"{type(inference_error).__name__}: {inference_error}"
            )
        elif not action_valid:
            command_allowed = False
            reason = "actor output is non-finite"
            raw_action = np.zeros(ACTION_DIM, dtype=np.float64)
        elif inference_time_ms > self.max_inference_time_ms:
            command_allowed = False
            reason = (
                f"inference timeout {inference_time_ms:.3f} ms "
                f"> {self.max_inference_time_ms:.3f} ms"
            )
        if update_action_history and action_valid and inference_error is None:
            self.observation_builder.set_previous_action(raw_action)
        self.sequence += 1

        if command_allowed:
            command = self.controller.policy_command(
                raw_action,
                sequence=self.sequence,
                armed=True,
                reason=reason,
                hardware_safe=hardware_safe,
                kp_scale=kp_scale,
                kd_scale=kd_scale,
            )
        else:
            command = self.controller.hold_command(
                self._last_finite_joint_pos,
                sequence=self.sequence,
                reason=reason,
                armed=False,
            )
        return PolicyStep(
            actor_obs=actor_obs,
            raw_action=raw_action,
            command=command,
            inference_time_ms=inference_time_ms,
        )
