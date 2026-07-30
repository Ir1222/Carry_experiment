"""PhysHSI CarryBox MuJoCo server for DDS or local UDP sim2sim."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from deploy.common.camera import (
    CameraIntrinsics,
    project_box_to_camera,
)
from deploy.common.config import load_deploy_config
from deploy.common.constants import DEFAULT_DOF_POS, KD, KP
from deploy.common.kinematics import (
    MujocoNameMap,
    task_state_from_mujoco,
)
from deploy.common.jsonl import JsonlRecorder
from deploy.common.mapping import RobotDescription
from deploy.common.math_utils import quat_rotate_inverse_wxyz
from deploy.common.transport import (
    UdpLatestReceiver,
    UdpPublisher,
    pack_robot_state,
    pack_task_state,
    unpack_policy_command,
)
from deploy.common.types import PolicyCommand
from deploy.tools.build_mjcf import build_robot_mjcf


def _require_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required; install deploy/requirements.txt"
        ) from exc
    return mujoco


def source_platform_center(
    box_position: np.ndarray,
    box_size: np.ndarray,
    platform_size: np.ndarray,
    box_gap: float,
) -> np.ndarray:
    """Place the fixed platform below a box using the Isaac default-reset gap."""

    box_position = np.asarray(box_position, dtype=np.float64)
    box_size = np.asarray(box_size, dtype=np.float64)
    platform_size = np.asarray(platform_size, dtype=np.float64)
    if box_position.shape != (3,):
        raise ValueError("box_position must have shape (3,)")
    if box_size.shape != (3,) or np.any(box_size <= 0.0):
        raise ValueError("box_size must contain three positive values")
    if platform_size.shape != (3,) or np.any(platform_size <= 0.0):
        raise ValueError("platform_size must contain three positive values")
    if not np.isfinite(box_gap) or box_gap < 0.0:
        raise ValueError("box_gap must be finite and non-negative")
    center = box_position.copy()
    center[2] -= 0.5 * box_size[2] + float(box_gap) + 0.5 * platform_size[2]
    return center


class MujocoServer:
    def __init__(
        self,
        config,
        *,
        transport: str,
        viewer: bool,
        camera_view: str = "free",
        log_path=None,
    ) -> None:
        self.cfg = config
        self.sim_cfg = config.section("simulation")
        self.control_cfg = config.section("control")
        self.network_cfg = config.section("network")
        self.robot_cfg = config.section("robot")
        self.camera_cfg = config.section("camera")
        self.mujoco = _require_mujoco()
        generated = config.resolve_path(self.sim_cfg["generated_robot_mjcf"])
        if not generated.exists():
            print(f"Generated robot MJCF not found; building {generated}")
            build_robot_mjcf(
                config.urdf_path,
                generated,
                joint_armature=float(self.sim_cfg.get("joint_armature", 0.01)),
                camera_config=self.camera_cfg,
            )
        self.model = self.mujoco.MjModel.from_xml_path(str(config.mjcf_path))
        self.data = self.mujoco.MjData(self.model)
        self.model.opt.timestep = float(self.sim_cfg["physics_dt"])
        self.robot = RobotDescription.from_urdf(config.urdf_path)
        self.name_map = MujocoNameMap(
            self.model,
            self.robot,
            pelvis_body=self.robot_cfg["base_body"],
            torso_body=self.robot_cfg["torso_body"],
            policy_frame_body=self.robot_cfg["policy_frame"],
            end_effector_names=tuple(self.robot_cfg["end_effectors"]),
        )
        if self.name_map.actuator_ids is None:
            raise RuntimeError("generated MJCF does not contain all 29 named actuators")
        self.transport_name = transport
        self.viewer_enabled = viewer
        if camera_view not in ("free", "d455"):
            raise ValueError(
                "camera_view must be either 'free' or 'd455'"
            )
        self.camera_view = camera_view
        self._viewer_camera_dirty = True
        self.recorder = JsonlRecorder(log_path)
        self.sequence = 0
        self.reset_requested = False
        self.stop_requested = False
        self._configure_transport()
        self._configure_scene()
        self._configure_camera()
        self.physics_fingerprint = self._configure_physics_profile()
        print(
            "MuJoCo physics profile: "
            + json.dumps(self.physics_fingerprint, sort_keys=True)
        )
        self.recorder.write(
            {
                "kind": "run_metadata",
                "component": "mujoco_server",
                "protocol_version": 2,
                "policy_frame": self.robot_cfg["policy_frame"],
                "physics": self.physics_fingerprint,
                "camera": self.camera_metadata,
            }
        )
        self.reset()

    @staticmethod
    def _address(value) -> tuple[str, int]:
        return str(value[0]), int(value[1])

    def _configure_transport(self) -> None:
        self.task_publisher = UdpPublisher(
            self._address(self.network_cfg["task_state_udp"])
        )
        if self.transport_name == "udp":
            self.robot_publisher = UdpPublisher(
                self._address(self.network_cfg["robot_state_udp"])
            )
            self.command_receiver = UdpLatestReceiver(
                self._address(self.network_cfg["robot_command_udp"]),
                unpack_policy_command,
            )
            self.unitree_bridge = None
        elif self.transport_name == "unitree_dds":
            from .unitree_bridge import UnitreeSimulatorBridge

            self.unitree_bridge = UnitreeSimulatorBridge(
                domain_id=int(self.network_cfg["domain_id"]),
                interface=str(self.network_cfg["interface"]),
                policy_to_motor=self.robot_cfg["policy_to_motor"],
            )
            self.robot_publisher = None
            self.command_receiver = None
        else:
            raise ValueError(f"unsupported transport {self.transport_name!r}")

    def _configure_scene(self) -> None:
        self.box_size = np.asarray(self.sim_cfg["box_size"], dtype=np.float64)
        self.box_initial_position = np.asarray(
            self.sim_cfg["box_initial_position"], dtype=np.float64
        )
        self.goal_position = np.asarray(
            self.sim_cfg["goal_position"], dtype=np.float64
        )
        self.box_body_id = int(self.model.body("carry_box").id)
        self.box_geom_id = int(self.model.geom("carry_box_geom").id)
        self.source_platform_body_id = int(
            self.model.body("source_platform").id
        )
        self.source_platform_geom_id = int(
            self.model.geom("source_platform_geom").id
        )
        self.goal_site_id = int(self.model.site("goal_site").id)
        self.head_body_id = int(self.model.body("mid360_link").id)
        self.hip_yaw_body_ids = np.asarray(
            [
                int(self.model.body("left_hip_yaw_link").id),
                int(self.model.body("right_hip_yaw_link").id),
            ],
            dtype=np.int32,
        )
        self.model.geom_size[self.box_geom_id, :3] = self.box_size / 2.0
        density = float(self.sim_cfg["box_density"])
        mass = density * float(np.prod(self.box_size))
        x, y, z = self.box_size
        self.model.body_mass[self.box_body_id] = mass
        self.model.body_inertia[self.box_body_id] = (
            mass
            / 12.0
            * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))
        )
        source_platform_cfg = self.sim_cfg["source_platform"]
        self.source_platform_enabled = bool(source_platform_cfg["enabled"])
        self.source_platform_size = np.asarray(
            source_platform_cfg["size"], dtype=np.float64
        )
        self.source_platform_box_gap = float(
            source_platform_cfg["box_gap"]
        )
        self.source_platform_friction = np.asarray(
            source_platform_cfg["friction"], dtype=np.float64
        )
        self.source_platform_position = source_platform_center(
            self.box_initial_position,
            self.box_size,
            self.source_platform_size,
            self.source_platform_box_gap,
        )
        self.model.geom_size[
            self.source_platform_geom_id, :3
        ] = self.source_platform_size / 2.0
        self.model.geom_friction[
            self.source_platform_geom_id, :3
        ] = self.source_platform_friction
        if self.source_platform_enabled:
            self.model.body_pos[
                self.source_platform_body_id
            ] = self.source_platform_position
            self.model.geom_contype[self.source_platform_geom_id] = 1
            self.model.geom_conaffinity[self.source_platform_geom_id] = 1
        else:
            self.model.body_pos[self.source_platform_body_id] = np.asarray(
                (
                    self.source_platform_position[0],
                    self.source_platform_position[1],
                    -5.0,
                ),
                dtype=np.float64,
            )
            self.model.geom_contype[self.source_platform_geom_id] = 0
            self.model.geom_conaffinity[self.source_platform_geom_id] = 0
        self.model.site_pos[self.goal_site_id] = self.goal_position

    def _configure_camera(self) -> None:
        camera_name = str(self.camera_cfg["name"])
        camera_body_name = str(self.camera_cfg["body"])
        try:
            self.camera_id = int(self.model.camera(camera_name).id)
            self.camera_body_id = int(
                self.model.body(camera_body_name).id
            )
        except KeyError as exc:
            raise RuntimeError(
                "generated MJCF does not contain the configured D455 "
                "camera; run `python -m deploy.tools.build_mjcf`"
            ) from exc
        actual_body_id = int(self.model.cam_bodyid[self.camera_id])
        if actual_body_id != self.camera_body_id:
            actual_body = self.model.body(actual_body_id).name
            raise RuntimeError(
                f"camera {camera_name!r} belongs to {actual_body!r}, "
                f"expected {camera_body_name!r}"
            )
        self.camera_intrinsics = CameraIntrinsics.from_config(
            self.camera_cfg
        )
        self.camera_metadata = {
            "name": camera_name,
            "body": camera_body_name,
            "position": [
                float(value) for value in self.camera_cfg["position"]
            ],
            "quaternion_wxyz": [
                float(value)
                for value in self.camera_cfg["quaternion_wxyz"]
            ],
            **self.camera_intrinsics.to_dict(),
        }

    def _validate_camera_axes(self) -> None:
        camera_rotation = self.data.cam_xmat[
            self.camera_id
        ].reshape(3, 3)
        optical_rotation = self.data.xmat[
            self.camera_body_id
        ].reshape(3, 3)
        camera_right = camera_rotation[:, 0]
        camera_up = camera_rotation[:, 1]
        camera_forward = -camera_rotation[:, 2]
        expected_right = optical_rotation[:, 0]
        expected_up = -optical_rotation[:, 1]
        expected_forward = optical_rotation[:, 2]
        if not (
            np.allclose(camera_right, expected_right, atol=1e-6)
            and np.allclose(camera_up, expected_up, atol=1e-6)
            and np.allclose(
                camera_forward, expected_forward, atol=1e-6
            )
        ):
            raise RuntimeError(
                "D455 camera axes do not match RealSense optical "
                "frame (+Z forward, +X right, +Y down)"
            )

    def box_camera_projection(self) -> dict[str, object]:
        return project_box_to_camera(
            camera_position_world=self.data.cam_xpos[
                self.camera_id
            ],
            camera_rotation_to_world=self.data.cam_xmat[
                self.camera_id
            ].reshape(3, 3),
            box_position_world=self.data.xpos[self.box_body_id],
            box_rotation_to_world=self.data.xmat[
                self.box_body_id
            ].reshape(3, 3),
            box_half_size=self.box_size / 2.0,
            intrinsics=self.camera_intrinsics,
        )

    def _configure_physics_profile(self) -> dict:
        contact_margin = float(self.sim_cfg.get("contact_margin", 0.0))
        collision_mask = (self.model.geom_contype != 0) | (
            self.model.geom_conaffinity != 0
        )
        self.model.geom_margin[collision_mask] = contact_margin

        solref = np.asarray(
            self.sim_cfg.get("joint_limit_solref", (0.005, 1.0)),
            dtype=np.float64,
        )
        limited = self.model.jnt_limited.astype(bool)
        self.model.jnt_solref[limited, :2] = solref
        solimp = np.asarray(
            self.sim_cfg.get(
                "joint_limit_solimp",
                (0.9, 0.95, 0.001, 0.5, 2.0),
            ),
            dtype=np.float64,
        )
        self.model.jnt_solimp[limited, :5] = solimp
        disable_refsafe = bool(
            self.sim_cfg.get("disable_refsafe_for_joint_limits", False)
        )
        if disable_refsafe:
            self.model.opt.disableflags |= int(
                self.mujoco.mjtDisableBit.mjDSBL_REFSAFE
            )

        base_joint_id = int(self.model.joint("floating_base_joint").id)
        base_dof_adr = int(self.model.jnt_dofadr[base_joint_id])
        linear_damping = float(
            self.sim_cfg.get("body_linear_damping", 0.0)
        )
        angular_damping = float(
            self.sim_cfg.get("body_angular_damping", 0.0)
        )
        self.model.dof_damping[base_dof_adr : base_dof_adr + 3] = (
            linear_damping
        )
        self.model.dof_damping[base_dof_adr + 3 : base_dof_adr + 6] = (
            angular_damping
        )
        return {
            "timestep": float(self.model.opt.timestep),
            "integrator": int(self.model.opt.integrator),
            "solver": int(self.model.opt.solver),
            "iterations": int(self.model.opt.iterations),
            "contact_margin": contact_margin,
            "collision_geom_count": int(np.count_nonzero(collision_mask)),
            "joint_limit_solref": solref.tolist(),
            "joint_limit_solimp": solimp.tolist(),
            "refsafe_disabled": disable_refsafe,
            "limited_joint_count": int(np.count_nonzero(limited)),
            "body_linear_damping": linear_damping,
            "body_angular_damping": angular_damping,
            "ground_friction": self.model.geom_friction[
                int(self.model.geom("floor").id)
            ].tolist(),
            "box_mass": float(self.model.body_mass[self.box_body_id]),
            "box_inertia": self.model.body_inertia[
                self.box_body_id
            ].tolist(),
            "source_platform": {
                "enabled": self.source_platform_enabled,
                "size": self.source_platform_size.tolist(),
                "position": self.source_platform_position.tolist(),
                "box_gap": self.source_platform_box_gap,
                "friction": self.source_platform_friction.tolist(),
            },
        }

    def _free_joint_address(self, name: str) -> int:
        return int(self.model.joint(name).qposadr)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        base_adr = self._free_joint_address("floating_base_joint")
        self.data.qpos[base_adr : base_adr + 3] = np.asarray(
            self.sim_cfg["robot_initial_position"], dtype=np.float64
        )
        self.data.qpos[base_adr + 3 : base_adr + 7] = np.asarray(
            self.sim_cfg["robot_initial_quaternion_wxyz"], dtype=np.float64
        )
        self.data.qpos[self.name_map.joint_qpos_adr] = np.asarray(DEFAULT_DOF_POS)
        box_adr = self._free_joint_address("box_free_joint")
        self.data.qpos[box_adr : box_adr + 3] = self.box_initial_position
        self.data.qpos[box_adr + 3 : box_adr + 7] = np.asarray(
            self.sim_cfg["box_initial_quaternion_wxyz"], dtype=np.float64
        )
        self.mujoco.mj_forward(self.model, self.data)
        self._validate_camera_axes()
        self.sequence = 0
        if self.unitree_bridge is not None:
            self.unitree_bridge.clear_command()
        elif self.command_receiver is not None:
            self.command_receiver.poll_latest()
        self.last_command = PolicyCommand(
            sequence=0,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(29),
            q_target=np.asarray(DEFAULT_DOF_POS),
            kp=np.asarray(KP),
            kd=np.asarray(KD),
            armed=False,
            reason="simulator default-pose hold",
        )
        self.has_received_command = False
        self.physics_started = False
        self.pending_command: PolicyCommand | None = None
        self.control_substep = 0
        self.control_decimation = int(self.control_cfg["decimation"])
        self.last_boundary_wait_ms = 0.0
        self._interval_max_gravity_xy = 0.0
        self._interval_max_joint_violation = 0.0
        self._interval_ground_contact_bodies: set[str] = set()
        self.episode_failed = False
        self.episode_failure_reason = ""
        self.episode_failure_sequence: int | None = None
        self.episode_failure_sim_time: float | None = None
        self.reset_requested = False

    def _poll_command(self) -> None:
        command = (
            self.command_receiver.poll_latest()
            if self.command_receiver is not None
            else self.unitree_bridge.poll_command()
        )
        if command is not None and command.is_finite():
            self.has_received_command = True
            if command.armed and self.episode_failed:
                pass
            elif command.armed:
                if not self.physics_started:
                    self.last_command = command
                    self.pending_command = None
                    self.control_substep = 0
                    # Paused-state publications may advance while the first
                    # action is in flight. Re-anchor the active physics
                    # sequence to the exact state used by that action so the
                    # next policy boundary is source_sequence + 4.
                    self.sequence = int(command.sequence)
                    self.physics_started = True
                else:
                    self.pending_command = command
            else:
                self.last_command = command
                self.pending_command = None
                self.control_substep = 0
                self.physics_started = False
        newest_command = (
            self.pending_command
            if self.pending_command is not None
            else self.last_command
        )
        age_ns = time.monotonic_ns() - newest_command.timestamp_ns
        if self.has_received_command and age_ns > int(0.2e9):
            joint_pos, _ = self.name_map.joint_state(self.data)
            self.last_command = PolicyCommand(
                sequence=self.last_command.sequence,
                timestamp_ns=time.monotonic_ns(),
                raw_action=np.zeros(29),
                q_target=joint_pos,
                kp=np.zeros(29),
                kd=np.asarray(KD),
                armed=False,
                reason="command timeout damping hold",
            )
            self.pending_command = None
            self.control_substep = 0
            self.physics_started = False

    @staticmethod
    def _roll_pitch_from_wxyz(quaternion: np.ndarray) -> tuple[float, float]:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(
            np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        )
        return float(roll), float(pitch)

    def _training_termination_reason(
        self,
        robot_state,
        projected_gravity: np.ndarray,
    ) -> str | None:
        gravity_xy = float(np.linalg.norm(projected_gravity[:2]))
        if gravity_xy > float(
            self.cfg.section("safety")["max_projected_gravity_xy"]
        ):
            return (
                "training projected-gravity termination: "
                f"xy={gravity_xy:.6f}"
            )
        head_height = float(self.data.xpos[self.head_body_id, 2])
        if head_height < float(
            self.sim_cfg.get("termination_head_height", 0.6)
        ):
            return (
                "training head-height termination: "
                f"z={head_height:.6f}"
            )
        root_height = float(
            self.data.xpos[self.name_map.pelvis_body_id, 2]
        )
        if root_height < float(
            self.sim_cfg.get("termination_root_height", 0.2)
        ):
            return (
                "training root-height termination: "
                f"z={root_height:.6f}"
            )
        hip_heights = self.data.xpos[self.hip_yaw_body_ids, 2]
        hip_min_index = int(np.argmin(hip_heights))
        hip_min = float(hip_heights[hip_min_index])
        if hip_min < float(
            self.sim_cfg.get("termination_hip_yaw_height", 0.15)
        ):
            body_name = self.model.body(
                int(self.hip_yaw_body_ids[hip_min_index])
            ).name
            return (
                "training hip-height termination: "
                f"body={body_name} z={hip_min:.6f}"
            )
        roll, pitch = self._roll_pitch_from_wxyz(
            robot_state.policy_frame_quat_wxyz
        )
        if abs(roll) > float(
            self.sim_cfg.get("termination_roll_abs", 0.5)
        ):
            return f"training roll termination: roll={roll:.6f}"
        if abs(pitch) > float(
            self.sim_cfg.get("termination_pitch_abs", 1.1)
        ):
            return f"training pitch termination: pitch={pitch:.6f}"
        return None

    def _latch_episode_failure(self, reason: str) -> None:
        if self.episode_failed:
            return
        self.episode_failed = True
        self.episode_failure_reason = reason
        self.episode_failure_sequence = self.sequence
        self.episode_failure_sim_time = float(self.data.time)
        joint_pos, _ = self.name_map.joint_state(self.data)
        self.last_command = PolicyCommand(
            sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(29),
            q_target=joint_pos,
            kp=np.zeros(29),
            kd=np.asarray(KD),
            armed=False,
            reason=f"episode failed: {reason}",
        )
        self.pending_command = None
        self.control_substep = 0
        self.physics_started = False

    def _apply_command(self) -> np.ndarray:
        joint_pos, joint_vel = self.name_map.joint_state(self.data)
        command = self.last_command
        torque = (
            command.tau_ff
            + command.kp * (command.q_target - joint_pos)
            - command.kd * joint_vel
        )
        torque = np.clip(
            torque, -self.robot.effort_limits, self.robot.effort_limits
        )
        self.data.ctrl[self.name_map.actuator_ids] = torque
        return torque

    def _wait_for_boundary_command(self) -> None:
        """Synchronize the next 4-step block with its boundary observation."""

        if not self.physics_started or self.control_substep != 0:
            self.last_boundary_wait_ms = 0.0
            return
        wait_start = time.perf_counter()
        required_state_sequence = self.sequence
        if (
            self.pending_command is None
            and self.last_command.sequence >= required_state_sequence
        ):
            self.last_boundary_wait_ms = 0.0
            return
        timeout_s = float(
            self.sim_cfg.get("policy_boundary_timeout_ms", 40.0)
        ) / 1000.0
        deadline = time.perf_counter() + timeout_s
        next_republish = time.perf_counter()
        while self.physics_started:
            if (
                self.pending_command is not None
                and self.pending_command.sequence >= required_state_sequence
            ):
                self.last_boundary_wait_ms = (
                    time.perf_counter() - wait_start
                ) * 1000.0
                return
            if time.perf_counter() >= deadline:
                joint_pos, _ = self.name_map.joint_state(self.data)
                self.last_command = PolicyCommand(
                    sequence=required_state_sequence,
                    timestamp_ns=time.monotonic_ns(),
                    raw_action=np.zeros(29),
                    q_target=joint_pos,
                    kp=np.zeros(29),
                    kd=np.asarray(KD),
                    armed=False,
                    reason=(
                        "policy boundary timeout waiting for state "
                        f"{required_state_sequence}"
                    ),
                )
                self.pending_command = None
                self.physics_started = False
                self.last_boundary_wait_ms = (
                    time.perf_counter() - wait_start
                ) * 1000.0
                return
            if time.perf_counter() >= next_republish:
                # UDP is intentionally best-effort. Re-publish the frozen
                # boundary with a fresh monotonic timestamp so one dropped
                # state/task packet cannot deadlock the 4-step handshake.
                self._publish(advance_sequence=False)
                next_republish = time.perf_counter() + 0.002
            time.sleep(0.0005)
            self._poll_command()

    def _publish(self, *, advance_sequence: bool = True):
        if advance_sequence:
            self.sequence += 1
        robot_state = self.name_map.robot_state(
            self.model, self.data, sequence=self.sequence
        )
        task_state = task_state_from_mujoco(
            self.model,
            self.data,
            self.name_map,
            sequence=self.sequence,
            box_body="carry_box",
            box_size=self.box_size,
            goal_position_world=self.goal_position,
            success_position_threshold=float(
                self.sim_cfg["success_position_threshold"]
            ),
            success_tilt_threshold=float(self.sim_cfg["success_tilt_threshold"]),
        )
        if self.robot_publisher is not None:
            self.robot_publisher.send(pack_robot_state(robot_state))
        else:
            self.unitree_bridge.publish_state(robot_state, self.data.time)
        self.task_publisher.send(pack_task_state(task_state))
        return robot_state, task_state

    def step(self) -> None:
        self._poll_command()
        self._wait_for_boundary_command()
        if self.physics_started:
            if self.control_substep == 0 and self.pending_command is not None:
                self.last_command = self.pending_command
                self.pending_command = None
            torque = self._apply_command()
            self.mujoco.mj_step(self.model, self.data)
            self.control_substep = (
                self.control_substep + 1
            ) % self.control_decimation
        else:
            torque = np.zeros(29, dtype=np.float64)
            self.data.ctrl[:] = 0.0
            self.data.time += float(self.model.opt.timestep)
            self.mujoco.mj_forward(self.model, self.data)
        robot_state, task_state = self._publish()
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_state.policy_frame_quat_wxyz,
            np.array([0.0, 0.0, -1.0]),
        )
        failure_reason = self._training_termination_reason(
            robot_state, projected_gravity
        )
        if failure_reason is not None:
            self._latch_episode_failure(failure_reason)
        if not self.recorder.enabled:
            return
        contacts = []
        ground_contact_bodies: set[str] = set()
        max_contact_force = 0.0
        detailed_log = (
            self.control_substep == 0
            if self.physics_started
            else self.sequence % self.control_decimation == 0
        )
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            force = None
            if detailed_log:
                force = np.zeros(6, dtype=np.float64)
                self.mujoco.mj_contactForce(
                    self.model, self.data, contact_index, force
                )
                max_contact_force = max(
                    max_contact_force, float(np.linalg.norm(force[:3]))
                )
            geom1 = (
                self.mujoco.mj_id2name(
                    self.model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1),
                )
                or int(contact.geom1)
            )
            geom2 = (
                self.mujoco.mj_id2name(
                    self.model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom2),
                )
                or int(contact.geom2)
            )
            body1 = self.model.body(
                int(self.model.geom_bodyid[int(contact.geom1)])
            ).name
            body2 = self.model.body(
                int(self.model.geom_bodyid[int(contact.geom2)])
            ).name
            if str(geom1).lower() == "floor":
                ground_contact_bodies.add(body2)
            if str(geom2).lower() == "floor":
                ground_contact_bodies.add(body1)
            if detailed_log:
                contacts.append(
                    {
                        "geom1": geom1,
                        "geom2": geom2,
                        "body1": body1,
                        "body2": body2,
                        "force_contact_frame": force,
                    }
                )
        joint_violation = np.maximum(
            np.maximum(
                self.robot.lower_limits - robot_state.joint_pos,
                robot_state.joint_pos - self.robot.upper_limits,
            ),
            0.0,
        )
        self._interval_max_gravity_xy = max(
            self._interval_max_gravity_xy,
            float(np.linalg.norm(projected_gravity[:2])),
        )
        self._interval_max_joint_violation = max(
            self._interval_max_joint_violation,
            float(np.max(joint_violation)),
        )
        self._interval_ground_contact_bodies.update(
            ground_contact_bodies
        )
        if not detailed_log:
            return
        camera_projection = self.box_camera_projection()
        record = {
            "kind": "mujoco_step",
            "wall_timestamp_ns": time.monotonic_ns(),
            "sim_time": self.data.time,
            "sequence": robot_state.sequence,
            "physics_step_stride": self.control_decimation,
            "projected_gravity": projected_gravity,
            "max_projected_gravity_xy_interval": (
                self._interval_max_gravity_xy
            ),
            "pelvis_position_world": self.data.xpos[
                self.name_map.pelvis_body_id
            ].copy(),
            "command_armed": self.last_command.armed,
            "command_reason": self.last_command.reason,
            "active_command_source_sequence": self.last_command.sequence,
            "pending_command_source_sequence": (
                None
                if self.pending_command is None
                else self.pending_command.sequence
            ),
            "control_substep": self.control_substep,
            "boundary_wait_ms": self.last_boundary_wait_ms,
            "physics_started": self.physics_started,
            "episode_failed": self.episode_failed,
            "episode_failure_reason": self.episode_failure_reason,
            "episode_failure_sequence": self.episode_failure_sequence,
            "episode_failure_sim_time": self.episode_failure_sim_time,
            "joint_limit_violation_max": (
                self._interval_max_joint_violation
            ),
            "box_position_world": self.data.xpos[self.box_body_id].copy(),
            "success": task_state.success,
            "contact_count": int(self.data.ncon),
            "max_contact_force": max_contact_force,
            "ground_contact_bodies": sorted(
                self._interval_ground_contact_bodies
            ),
            "detail": detailed_log,
            "camera_view": self.camera_view,
            **camera_projection,
        }
        if detailed_log:
            record.update(
                {
                    "joint_pos": robot_state.joint_pos,
                    "joint_vel": robot_state.joint_vel,
                    "policy_frame_quat_wxyz": (
                        robot_state.policy_frame_quat_wxyz
                    ),
                    "policy_frame_ang_vel": (
                        robot_state.policy_frame_ang_vel
                    ),
                    "raw_action": self.last_command.raw_action,
                    "q_target": self.last_command.q_target,
                    "torque": torque,
                    "joint_limit_violation": joint_violation,
                    "box_pos_policy_frame": (
                        task_state.box_pos_policy_frame
                    ),
                    "box_quat_policy_frame_wxyz": (
                        task_state.box_quat_policy_frame_wxyz
                    ),
                    "goal_pos_policy_frame": (
                        task_state.goal_pos_policy_frame
                    ),
                    "contacts": contacts,
                }
            )
        self.recorder.write(record)
        self._interval_max_gravity_xy = 0.0
        self._interval_max_joint_violation = 0.0
        self._interval_ground_contact_bodies.clear()

    def _key_callback(self, keycode: int) -> None:
        if keycode == 259:  # GLFW_BACKSPACE
            self.reset_requested = True
        elif keycode == 67:  # GLFW_C
            self.camera_view = (
                "free" if self.camera_view == "d455" else "d455"
            )
            self._viewer_camera_dirty = True
            print(f"MuJoCo viewer camera: {self.camera_view}")
        elif keycode in (81, 256):  # Q or ESC
            self.stop_requested = True

    def _apply_viewer_camera(self, viewer_context) -> None:
        if not self._viewer_camera_dirty:
            return
        with viewer_context.lock():
            if self.camera_view == "d455":
                viewer_context.cam.type = (
                    self.mujoco.mjtCamera.mjCAMERA_FIXED
                )
                viewer_context.cam.fixedcamid = self.camera_id
            else:
                viewer_context.cam.type = (
                    self.mujoco.mjtCamera.mjCAMERA_FREE
                )
                viewer_context.cam.fixedcamid = -1
        self._viewer_camera_dirty = False

    @staticmethod
    def _wait_until(deadline: float) -> None:
        """Wait accurately enough for 200 Hz on Linux and Windows hosts."""

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return
            if remaining > 0.0015:
                time.sleep(remaining - 0.001)
            else:
                # A short yield/spin tail avoids accumulating millisecond
                # scheduler oversleep at every 5 ms physics tick.
                time.sleep(0)

    def run(self, *, duration: float | None = None) -> None:
        period = float(self.sim_cfg["physics_dt"])
        next_tick = time.perf_counter()
        start_time = next_tick
        if self.viewer_enabled:
            viewer_context = self.mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            self._apply_viewer_camera(viewer_context)
        else:
            viewer_context = None
        try:
            while not self.stop_requested:
                if duration is not None and time.perf_counter() - start_time >= duration:
                    break
                if viewer_context is not None and not viewer_context.is_running():
                    break
                if bool(self.sim_cfg.get("realtime", True)):
                    self._wait_until(next_tick)
                if self.reset_requested:
                    self.reset()
                self.step()
                if viewer_context is not None:
                    self._apply_viewer_camera(viewer_context)
                    viewer_context.sync()
                if bool(self.sim_cfg.get("realtime", True)):
                    next_tick += period
                    if next_tick < time.perf_counter() - 1.0:
                        next_tick = time.perf_counter()
        finally:
            if viewer_context is not None:
                viewer_context.close()
            self.close()

    def close(self) -> None:
        self.recorder.close()
        self.task_publisher.close()
        if self.robot_publisher is not None:
            self.robot_publisher.close()
        if self.command_receiver is not None:
            self.command_receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--transport", choices=("unitree_dds", "udp"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-view",
        choices=("free", "d455"),
        default="free",
        help="initial MuJoCo viewer camera; press C to toggle",
    )
    parser.add_argument("--log", help="optional per-physics-step JSONL log")
    parser.add_argument("--duration", type=float, help="optional wall-clock duration")
    args = parser.parse_args()
    cfg = load_deploy_config(args.config)
    server = MujocoServer(
        cfg,
        transport=args.transport or cfg.section("simulation")["transport"],
        viewer=bool(cfg.section("simulation")["viewer"]) and not args.headless,
        camera_view=args.camera_view,
        log_path=args.log,
    )
    server.run(duration=args.duration)


if __name__ == "__main__":
    main()
