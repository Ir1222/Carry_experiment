"""MuJoCo-side Unitree DDS LowState/LowCmd bridge."""

from __future__ import annotations

import threading
import time

import numpy as np

from deploy.common.constants import ACTION_DIM
from deploy.common.mapping import validate_motor_mapping
from deploy.common.types import PolicyCommand, RobotState


class UnitreeSimulatorBridge:
    def __init__(
        self,
        *,
        domain_id: int,
        interface: str,
        policy_to_motor,
    ) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__LowState_,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "Unitree SDK2 Python is required for unitree_dds transport"
            ) from exc

        ChannelFactoryInitialize(int(domain_id), str(interface))
        self.mapping = validate_motor_mapping(policy_to_motor)
        self._state = unitree_hg_msg_dds__LowState_()
        self._publisher = ChannelPublisher("rt/lowstate", LowState_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._subscriber.Init(self._low_command_handler, 1)
        self._crc = CRC()
        self._latest_command: PolicyCommand | None = None
        self._lock = threading.Lock()
        self._command_sequence = 0

    def _low_command_handler(self, message) -> None:
        q_target = np.zeros(ACTION_DIM)
        kp = np.zeros(ACTION_DIM)
        kd = np.zeros(ACTION_DIM)
        tau = np.zeros(ACTION_DIM)
        for policy_index, motor_index in enumerate(self.mapping):
            motor = message.motor_cmd[int(motor_index)]
            q_target[policy_index] = motor.q
            kp[policy_index] = motor.kp
            kd[policy_index] = motor.kd
            tau[policy_index] = motor.tau
        self._command_sequence += 1
        command = PolicyCommand(
            sequence=self._command_sequence,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(ACTION_DIM),
            q_target=q_target,
            kp=kp,
            kd=kd,
            tau_ff=tau,
            # Policy-active commands carry stiffness. A zero-Kp damping hold
            # must not release the simulator's pre-arm frozen reset state.
            armed=bool(np.any(kp > 0.0)),
            reason="Unitree LowCmd",
        )
        with self._lock:
            self._latest_command = command

    def poll_command(self) -> PolicyCommand | None:
        with self._lock:
            return self._latest_command

    def clear_command(self) -> None:
        with self._lock:
            self._latest_command = None

    def publish_state(self, state: RobotState, sim_time: float) -> None:
        low_state = self._state
        for policy_index, motor_index in enumerate(self.mapping):
            motor = low_state.motor_state[int(motor_index)]
            motor.q = float(state.joint_pos[policy_index])
            motor.dq = float(state.joint_vel[policy_index])
        low_state.imu_state.quaternion[:] = state.torso_quat_wxyz
        low_state.imu_state.gyroscope[:] = state.torso_ang_vel
        low_state.tick = int(sim_time * 1e3)
        low_state.crc = self._crc.Crc(low_state)
        self._publisher.Write(low_state)
