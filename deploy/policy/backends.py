"""Policy-side state/command backend factories."""

from __future__ import annotations

from deploy.common.transport import (
    UdpLatestReceiver,
    UdpPublisher,
    pack_policy_command,
    unpack_robot_state,
    unpack_task_state,
)
from deploy.common.types import PolicyCommand, RobotState, TaskState


class UdpPolicyBackend:
    def __init__(
        self,
        robot_state_address: tuple[str, int],
        task_state_address: tuple[str, int],
        command_address: tuple[str, int],
    ) -> None:
        self.robot_receiver = UdpLatestReceiver(
            robot_state_address, unpack_robot_state
        )
        self.task_receiver = UdpLatestReceiver(task_state_address, unpack_task_state)
        self.command_publisher = UdpPublisher(command_address)
        self._robot: RobotState | None = None
        self._task: TaskState | None = None

    def poll(self) -> tuple[RobotState | None, TaskState | None]:
        robot = self.robot_receiver.poll_latest()
        task = self.task_receiver.poll_latest()
        if robot is not None:
            self._robot = robot
        if task is not None:
            self._task = task
        return self._robot, self._task

    def send(self, command: PolicyCommand) -> None:
        self.command_publisher.send(pack_policy_command(command))

    def close(self) -> None:
        self.robot_receiver.close()
        self.task_receiver.close()
        self.command_publisher.close()
