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


class SequenceStatePair:
    """Keep the newest robot/task pair with exactly matching source sequence."""

    def __init__(self) -> None:
        self.robot: RobotState | None = None
        self.task: TaskState | None = None
        self.synchronized: tuple[
            RobotState | None, TaskState | None
        ] = (None, None)

    def update(
        self,
        robot: RobotState | None = None,
        task: TaskState | None = None,
    ) -> tuple[RobotState | None, TaskState | None]:
        if robot is not None:
            self.robot = robot
        if task is not None:
            self.task = task
        if (
            self.robot is not None
            and self.task is not None
            and self.robot.sequence == self.task.sequence
        ):
            self.synchronized = (self.robot, self.task)
        return self.synchronized


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
        self._state_pair = SequenceStatePair()

    def poll(self) -> tuple[RobotState | None, TaskState | None]:
        robot = self.robot_receiver.poll_latest()
        task = self.task_receiver.poll_latest()
        return self._state_pair.update(robot, task)

    def send(self, command: PolicyCommand) -> None:
        self.command_publisher.send(pack_policy_command(command))

    def close(self) -> None:
        self.robot_receiver.close()
        self.task_receiver.close()
        self.command_publisher.close()
