"""Task-state provider interface and UDP implementation."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from deploy.common.transport import UdpLatestReceiver, unpack_task_state
from deploy.common.types import TaskState


class TaskStateProvider(Protocol):
    def poll(self) -> TaskState | None: ...

    def close(self) -> None: ...


class UdpTaskStateProvider:
    def __init__(self, address: tuple[str, int]) -> None:
        self.receiver = UdpLatestReceiver(address, unpack_task_state)
        self.latest: TaskState | None = None

    def poll(self) -> TaskState | None:
        state = self.receiver.poll_latest()
        if state is not None:
            self.latest = state
        return self.latest

    def close(self) -> None:
        self.receiver.close()


class MockTaskStateProvider:
    """In-process deterministic perception substitute for dry-run tests."""

    def __init__(
        self,
        *,
        box_pos_policy_frame=(1.0, 0.0, -0.65),
        box_quat_policy_frame_wxyz=(1.0, 0.0, 0.0, 0.0),
        box_size=(0.3, 0.3, 0.25),
        goal_pos_policy_frame=(2.5, 0.75, -0.65),
        success: bool = False,
    ) -> None:
        self.box_pos_policy_frame = np.asarray(
            box_pos_policy_frame, dtype=np.float64
        )
        self.box_quat_policy_frame_wxyz = np.asarray(
            box_quat_policy_frame_wxyz, dtype=np.float64
        )
        self.box_size = np.asarray(box_size, dtype=np.float64)
        self.goal_pos_policy_frame = np.asarray(
            goal_pos_policy_frame, dtype=np.float64
        )
        self.success = bool(success)
        self.sequence = 0

    def poll(self) -> TaskState:
        self.sequence += 1
        return TaskState(
            sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            box_pos_policy_frame=self.box_pos_policy_frame,
            box_quat_policy_frame_wxyz=self.box_quat_policy_frame_wxyz,
            box_size=self.box_size,
            goal_pos_policy_frame=self.goal_pos_policy_frame,
            success=self.success,
        )

    def close(self) -> None:
        return None
