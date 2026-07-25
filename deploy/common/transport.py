"""Versioned UDP packets for local sim2sim and task-state transport."""

from __future__ import annotations

import socket
import struct
from typing import Generic, TypeVar

import numpy as np

from .constants import ACTION_DIM
from .types import PolicyCommand, RobotState, TaskState

VERSION = 1
KIND_ROBOT = 1
KIND_TASK = 2
KIND_COMMAND = 3
FLAG_ACTIVE = 1
HEADER = struct.Struct("<4sBBHQQ")
MAGIC = b"PHSI"

ROBOT_FLOATS = 4 + 3 + ACTION_DIM + ACTION_DIM + 15
TASK_FLOATS = 3 + 4 + 3 + 3
COMMAND_FLOATS = ACTION_DIM * 5


def _pack(kind: int, sequence: int, timestamp_ns: int, values, flags: int = 0) -> bytes:
    array = np.asarray(values, dtype="<f4").reshape(-1)
    return HEADER.pack(MAGIC, VERSION, kind, flags, sequence, timestamp_ns) + array.tobytes()


def _unpack(payload: bytes, expected_kind: int, expected_floats: int):
    expected_size = HEADER.size + expected_floats * 4
    if len(payload) != expected_size:
        raise ValueError(f"packet has {len(payload)} bytes, expected {expected_size}")
    magic, version, kind, flags, sequence, timestamp_ns = HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION or kind != expected_kind:
        raise ValueError(
            f"invalid packet header magic={magic!r} version={version} kind={kind}"
        )
    values = np.frombuffer(payload, dtype="<f4", offset=HEADER.size).astype(np.float64)
    return sequence, timestamp_ns, flags, values


def pack_robot_state(state: RobotState) -> bytes:
    values = np.concatenate(
        (
            state.torso_quat_wxyz,
            state.torso_ang_vel,
            state.joint_pos,
            state.joint_vel,
            state.end_effector_pos_torso.reshape(-1),
        )
    )
    return _pack(KIND_ROBOT, state.sequence, state.timestamp_ns, values)


def unpack_robot_state(payload: bytes) -> RobotState:
    sequence, timestamp_ns, _, values = _unpack(
        payload, KIND_ROBOT, ROBOT_FLOATS
    )
    return RobotState(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        torso_quat_wxyz=values[0:4],
        torso_ang_vel=values[4:7],
        joint_pos=values[7:36],
        joint_vel=values[36:65],
        end_effector_pos_torso=values[65:80],
    )


def pack_task_state(state: TaskState) -> bytes:
    values = np.concatenate(
        (
            state.box_pos_torso,
            state.box_quat_torso_wxyz,
            state.box_size,
            state.goal_pos_torso,
        )
    )
    flags = FLAG_ACTIVE if state.success else 0
    return _pack(KIND_TASK, state.sequence, state.timestamp_ns, values, flags)


def unpack_task_state(payload: bytes) -> TaskState:
    sequence, timestamp_ns, flags, values = _unpack(
        payload, KIND_TASK, TASK_FLOATS
    )
    return TaskState(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        box_pos_torso=values[0:3],
        box_quat_torso_wxyz=values[3:7],
        box_size=values[7:10],
        goal_pos_torso=values[10:13],
        success=bool(flags & FLAG_ACTIVE),
    )


def pack_policy_command(command: PolicyCommand) -> bytes:
    values = np.concatenate(
        (
            command.raw_action,
            command.q_target,
            command.kp,
            command.kd,
            command.tau_ff,
        )
    )
    flags = FLAG_ACTIVE if command.armed else 0
    return _pack(
        KIND_COMMAND,
        command.sequence,
        command.timestamp_ns,
        values,
        flags,
    )


def unpack_policy_command(payload: bytes) -> PolicyCommand:
    sequence, timestamp_ns, flags, values = _unpack(
        payload, KIND_COMMAND, COMMAND_FLOATS
    )
    return PolicyCommand(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        raw_action=values[0:29],
        q_target=values[29:58],
        kp=values[58:87],
        kd=values[87:116],
        tau_ff=values[116:145],
        armed=bool(flags & FLAG_ACTIVE),
        reason="received over UDP",
    )


T = TypeVar("T")


class UdpPublisher:
    def __init__(self, address: tuple[str, int]) -> None:
        self.address = (str(address[0]), int(address[1]))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: bytes) -> None:
        self.socket.sendto(payload, self.address)

    def close(self) -> None:
        self.socket.close()


class UdpLatestReceiver(Generic[T]):
    def __init__(self, address: tuple[str, int], decoder) -> None:
        self.address = (str(address[0]), int(address[1]))
        self.decoder = decoder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.address)
        self.socket.setblocking(False)
        self._last_timestamp_ns = -1

    def poll_latest(self) -> T | None:
        latest: T | None = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                decoded = self.decoder(payload)
            except (ValueError, struct.error):
                continue
            timestamp_ns = int(
                getattr(decoded, "timestamp_ns", self._last_timestamp_ns + 1)
            )
            if timestamp_ns <= self._last_timestamp_ns:
                continue
            self._last_timestamp_ns = timestamp_ns
            latest = decoded
        return latest

    def close(self) -> None:
        self.socket.close()
