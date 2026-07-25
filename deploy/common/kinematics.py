"""MuJoCo-backed name-based kinematics for simulation and real-state processing."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .constants import END_EFFECTOR_NAMES, JOINT_NAMES
from .mapping import RobotDescription
from .math_utils import (
    quat_relative_wxyz,
    quat_rotate_inverse_wxyz,
)
from .types import RobotState, TaskState


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for kinematics/simulation; "
            "install deploy/requirements.txt"
        ) from exc
    return mujoco


class MujocoNameMap:
    """Resolve all qpos/qvel/body/actuator addresses by name exactly once."""

    def __init__(
        self,
        model,
        robot: RobotDescription,
        *,
        pelvis_body: str = "pelvis",
        torso_body: str = "torso_link",
        end_effector_names: tuple[str, ...] = END_EFFECTOR_NAMES,
    ) -> None:
        mujoco = _require_mujoco()
        runtime_joints = []
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name in JOINT_NAMES:
                runtime_joints.append(name)
        robot.assert_runtime_names(runtime_joints)

        self.joint_qpos_adr = np.asarray(
            [int(model.joint(name).qposadr[0]) for name in robot.joint_names],
            dtype=np.int64,
        )
        self.joint_dof_adr = np.asarray(
            [int(model.joint(name).dofadr[0]) for name in robot.joint_names],
            dtype=np.int64,
        )
        self.pelvis_body_id = int(model.body(pelvis_body).id)
        self.torso_body_id = int(model.body(torso_body).id)
        self.end_effector_body_ids = np.asarray(
            [model.body(name).id for name in end_effector_names], dtype=np.int64
        )
        if len(set(self.end_effector_body_ids.tolist())) != len(end_effector_names):
            raise ValueError("end-effector body mapping is not one-to-one")

        actuator_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index): index
            for index in range(model.nu)
        }
        missing_actuators = [name for name in robot.joint_names if name not in actuator_names]
        self.actuator_ids = (
            None
            if missing_actuators
            else np.asarray(
                [actuator_names[name] for name in robot.joint_names], dtype=np.int64
            )
        )

    def joint_state(self, data) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(data.qpos[self.joint_qpos_adr], dtype=np.float64).copy(),
            np.asarray(data.qvel[self.joint_dof_adr], dtype=np.float64).copy(),
        )

    def endpoint_positions_torso(self, data) -> np.ndarray:
        pelvis_position = np.asarray(
            data.xpos[self.pelvis_body_id], dtype=np.float64
        )
        torso_quat = np.asarray(data.xquat[self.torso_body_id], dtype=np.float64)
        endpoint_world = np.asarray(
            data.xpos[self.end_effector_body_ids], dtype=np.float64
        )
        return np.stack(
            [
                quat_rotate_inverse_wxyz(torso_quat, position - pelvis_position)
                for position in endpoint_world
            ],
            axis=0,
        )

    def torso_angular_velocity(self, model, data) -> np.ndarray:
        mujoco = _require_mujoco()
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body_id,
            velocity,
            1,
        )
        return velocity[0:3].copy()

    def robot_state(self, model, data, *, sequence: int) -> RobotState:
        joint_pos, joint_vel = self.joint_state(data)
        return RobotState(
            sequence=sequence,
            timestamp_ns=time.monotonic_ns(),
            torso_quat_wxyz=np.asarray(
                data.xquat[self.torso_body_id], dtype=np.float64
            ).copy(),
            torso_ang_vel=self.torso_angular_velocity(model, data),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            end_effector_pos_torso=self.endpoint_positions_torso(data),
        )


class MujocoKinematicsProvider:
    """Forward kinematics from only the 29 measured joint positions."""

    def __init__(
        self,
        mjcf_path: str | Path,
        robot: RobotDescription,
        *,
        pelvis_body: str = "pelvis",
        torso_body: str = "torso_link",
        end_effector_names: tuple[str, ...] = END_EFFECTOR_NAMES,
    ) -> None:
        mujoco = _require_mujoco()
        self.model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.name_map = MujocoNameMap(
            self.model,
            robot,
            pelvis_body=pelvis_body,
            torso_body=torso_body,
            end_effector_names=end_effector_names,
        )
        self._free_joint_qpos_adr: int | None = None
        for index in range(self.model.njnt):
            if self.model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, index
                )
                if name != "box_free_joint":
                    self._free_joint_qpos_adr = int(self.model.jnt_qposadr[index])
                    break

    def endpoints(self, joint_pos: np.ndarray) -> np.ndarray:
        self._set_joint_state(joint_pos)
        mujoco = _require_mujoco()
        mujoco.mj_forward(self.model, self.data)
        return self.name_map.endpoint_positions_torso(self.data)

    def _set_joint_state(
        self, joint_pos: np.ndarray, joint_vel: np.ndarray | None = None
    ) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if joint_pos.shape != (len(JOINT_NAMES),):
            raise ValueError("joint_pos must have shape (29,)")
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self._free_joint_qpos_adr is not None:
            adr = self._free_joint_qpos_adr
            self.data.qpos[adr + 3] = 1.0
        self.data.qpos[self.name_map.joint_qpos_adr] = joint_pos
        if joint_vel is not None:
            joint_vel = np.asarray(joint_vel, dtype=np.float64)
            if joint_vel.shape != (len(JOINT_NAMES),):
                raise ValueError("joint_vel must have shape (29,)")
            self.data.qvel[self.name_map.joint_dof_adr] = joint_vel

    def torso_relative_state(
        self, joint_pos: np.ndarray, joint_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pelvis-to-torso quaternion and relative gyro in torso axes."""
        mujoco = _require_mujoco()
        self._set_joint_state(joint_pos, joint_vel)
        mujoco.mj_forward(self.model, self.data)
        torso_quat = np.asarray(
            self.data.xquat[self.name_map.torso_body_id], dtype=np.float64
        ).copy()
        relative_gyro = self.name_map.torso_angular_velocity(
            self.model, self.data
        )
        return torso_quat, relative_gyro


def task_state_from_mujoco(
    model,
    data,
    name_map: MujocoNameMap,
    *,
    sequence: int,
    box_body: str,
    box_size: np.ndarray,
    goal_position_world: np.ndarray,
    success_position_threshold: float,
    success_tilt_threshold: float,
) -> TaskState:
    box_id = int(model.body(box_body).id)
    pelvis_position = np.asarray(
        data.xpos[name_map.pelvis_body_id], dtype=np.float64
    )
    torso_quat = np.asarray(data.xquat[name_map.torso_body_id], dtype=np.float64)
    box_position = np.asarray(data.xpos[box_id], dtype=np.float64)
    box_quat = np.asarray(data.xquat[box_id], dtype=np.float64)
    box_pos_torso = quat_rotate_inverse_wxyz(
        torso_quat, box_position - pelvis_position
    )
    goal_position_world = np.asarray(goal_position_world, dtype=np.float64)
    goal_pos_torso = quat_rotate_inverse_wxyz(
        torso_quat, goal_position_world - pelvis_position
    )
    box_quat_torso = quat_relative_wxyz(torso_quat, box_quat)
    box_gravity = quat_rotate_inverse_wxyz(
        box_quat, np.array([0.0, 0.0, -1.0])
    )
    success = (
        np.linalg.norm(box_position - goal_position_world)
        < float(success_position_threshold)
        and np.linalg.norm(box_gravity[:2]) < float(success_tilt_threshold)
    )
    return TaskState(
        sequence=sequence,
        timestamp_ns=time.monotonic_ns(),
        box_pos_torso=box_pos_torso,
        box_quat_torso_wxyz=box_quat_torso,
        box_size=np.asarray(box_size, dtype=np.float64),
        goal_pos_torso=goal_pos_torso,
        success=success,
    )
