"""Unitree SDK2 policy-side backend.

Imports are intentionally lazy so exporter/tests work without Unitree SDK2.
Real writes remain disabled unless the runner explicitly enables them.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from deploy.common.constants import ACTION_DIM, KD
from deploy.common.kinematics import MujocoKinematicsProvider
from deploy.common.mapping import validate_motor_mapping
from deploy.common.math_utils import (
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_rotate_wxyz,
)
from deploy.common.types import PolicyCommand, RobotState, TaskState
from deploy.sim2real.task_provider import (
    TaskStateProvider,
    UdpTaskStateProvider,
)


class UnitreePolicyBackend:
    def __init__(
        self,
        *,
        domain_id: int,
        interface: str,
        policy_to_motor,
        kinematics: MujocoKinematicsProvider,
        task_address: tuple[str, int],
        task_provider: TaskStateProvider | None = None,
        command_hz: float,
        write_enabled: bool,
        imu_frame: str,
        command_stale_timeout_ms: float,
        mode_machine: int = 5,
        mode_pr: int = 0,
    ) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "Unitree SDK2 Python is required for unitree_dds transport"
            ) from exc

        ChannelFactoryInitialize(int(domain_id), str(interface))
        self.mapping = validate_motor_mapping(policy_to_motor)
        self.kinematics = kinematics
        self.task_provider = (
            UdpTaskStateProvider(task_address)
            if task_provider is None
            else task_provider
        )
        self._latest_robot: RobotState | None = None
        self._last_finite_joint_pos: np.ndarray | None = None
        self._latest_task: TaskState | None = None
        self._sequence = 0
        self._last_tick: int | None = None
        self._lock = threading.Lock()
        self._write_enabled = bool(write_enabled)
        self._imu_frame = str(imu_frame).lower()
        if self._imu_frame not in ("torso", "pelvis"):
            raise ValueError("robot.imu_frame must be 'torso' or 'pelvis'")
        self._command_stale_timeout_ns = int(
            float(command_stale_timeout_ms) * 1e6
        )
        self._mode_machine = int(mode_machine)
        self._mode_pr = int(mode_pr)
        self._crc = CRC()
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._low_state_handler, 1)
        self._latest_command: PolicyCommand | None = None
        self._stop = threading.Event()
        self._period = 1.0 / float(command_hz)
        self._thread = threading.Thread(
            target=self._command_loop, name="unitree-command-loop", daemon=True
        )
        self._thread.start()

    def _low_state_handler(self, message) -> None:
        now = time.monotonic_ns()
        joint_pos = np.zeros(ACTION_DIM)
        joint_vel = np.zeros(ACTION_DIM)
        for policy_index, motor_index in enumerate(self.mapping):
            motor = message.motor_state[int(motor_index)]
            joint_pos[policy_index] = motor.q
            joint_vel[policy_index] = motor.dq
        endpoints = self.kinematics.endpoints(joint_pos)
        imu_quat = np.asarray(
            message.imu_state.quaternion, dtype=np.float64
        ).reshape(4)
        imu_gyro = np.asarray(message.imu_state.gyroscope, dtype=np.float64)
        if self._imu_frame == "torso":
            norm = float(np.linalg.norm(imu_quat))
            if np.isfinite(norm) and abs(norm - 1.0) <= 1e-3:
                pelvis_to_torso, relative_gyro = (
                    self.kinematics.torso_relative_state(joint_pos, joint_vel)
                )
                imu_quat = normalize_quat_wxyz(
                    quat_multiply_wxyz(
                        imu_quat,
                        quat_conjugate_wxyz(pelvis_to_torso),
                    )
                )
                imu_gyro = quat_rotate_wxyz(
                    pelvis_to_torso,
                    imu_gyro - relative_gyro,
                )
        tick = int(getattr(message, "tick", 0))
        if self._last_tick is not None and tick < self._last_tick:
            self._sequence = 0
        self._last_tick = tick
        self._sequence += 1
        state = RobotState(
            sequence=self._sequence,
            timestamp_ns=now,
            policy_frame_quat_wxyz=imu_quat,
            policy_frame_ang_vel=imu_gyro,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            end_effector_pos_policy_frame=endpoints,
        )
        with self._lock:
            self._latest_robot = state
            if np.isfinite(joint_pos).all():
                self._last_finite_joint_pos = joint_pos.copy()

    def poll(self) -> tuple[RobotState | None, TaskState | None]:
        task = self.task_provider.poll()
        with self._lock:
            if task is not None:
                self._latest_task = task
            return self._latest_robot, self._latest_task

    def send(self, command: PolicyCommand) -> None:
        if not command.is_finite():
            return
        with self._lock:
            self._latest_command = command

    def _fill_command(self, command: PolicyCommand) -> None:
        low_cmd = self._low_cmd
        if hasattr(low_cmd, "mode_machine"):
            low_cmd.mode_machine = self._mode_machine
        if hasattr(low_cmd, "mode_pr"):
            low_cmd.mode_pr = self._mode_pr
        for policy_index, motor_index in enumerate(self.mapping):
            motor = low_cmd.motor_cmd[int(motor_index)]
            motor.mode = 0x0A
            motor.q = float(command.q_target[policy_index])
            motor.dq = 0.0
            motor.kp = float(command.kp[policy_index])
            motor.kd = float(command.kd[policy_index])
            motor.tau = float(command.tau_ff[policy_index])
        low_cmd.crc = self._crc.Crc(low_cmd)

    def _command_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                command = self._latest_command
                last_finite_joint_pos = (
                    None
                    if self._last_finite_joint_pos is None
                    else self._last_finite_joint_pos.copy()
                )
            if (
                command is not None
                and time.monotonic_ns() - command.timestamp_ns
                > self._command_stale_timeout_ns
            ):
                if last_finite_joint_pos is None:
                    command = None
                else:
                    command = PolicyCommand(
                        sequence=command.sequence,
                        timestamp_ns=time.monotonic_ns(),
                        raw_action=np.zeros(ACTION_DIM),
                        q_target=last_finite_joint_pos,
                        kp=np.zeros(ACTION_DIM),
                        kd=np.asarray(KD, dtype=np.float64),
                        armed=False,
                        reason="policy command stale: damping hold",
                    )
            if self._write_enabled and command is not None:
                self._fill_command(command)
                self._publisher.Write(self._low_cmd)
            next_tick += self._period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.perf_counter()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.task_provider.close()
