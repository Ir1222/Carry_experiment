"""PhysHSI CarryBox MuJoCo server for DDS or local UDP sim2sim."""

from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.common.config import load_deploy_config
from deploy.common.constants import DEFAULT_DOF_POS, KD, KP
from deploy.common.kinematics import (
    MujocoNameMap,
    task_state_from_mujoco,
)
from deploy.common.jsonl import JsonlRecorder
from deploy.common.mapping import RobotDescription
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


class MujocoServer:
    def __init__(
        self, config, *, transport: str, viewer: bool, log_path=None
    ) -> None:
        self.cfg = config
        self.sim_cfg = config.section("simulation")
        self.control_cfg = config.section("control")
        self.network_cfg = config.section("network")
        self.robot_cfg = config.section("robot")
        self.mujoco = _require_mujoco()
        generated = config.resolve_path(self.sim_cfg["generated_robot_mjcf"])
        if not generated.exists():
            print(f"Generated robot MJCF not found; building {generated}")
            build_robot_mjcf(
                config.urdf_path,
                generated,
                joint_armature=float(self.sim_cfg.get("joint_armature", 0.01)),
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
            end_effector_names=tuple(self.robot_cfg["end_effectors"]),
        )
        if self.name_map.actuator_ids is None:
            raise RuntimeError("generated MJCF does not contain all 29 named actuators")
        self.transport_name = transport
        self.viewer_enabled = viewer
        self.recorder = JsonlRecorder(log_path)
        self.sequence = 0
        self.reset_requested = False
        self.stop_requested = False
        self._configure_transport()
        self._configure_scene()
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
        self.goal_position = np.asarray(
            self.sim_cfg["goal_position"], dtype=np.float64
        )
        self.box_body_id = int(self.model.body("carry_box").id)
        self.box_geom_id = int(self.model.geom("carry_box_geom").id)
        self.goal_site_id = int(self.model.site("goal_site").id)
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
        self.model.site_pos[self.goal_site_id] = self.goal_position

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
        self.data.qpos[box_adr : box_adr + 3] = np.asarray(
            self.sim_cfg["box_initial_position"], dtype=np.float64
        )
        self.data.qpos[box_adr + 3 : box_adr + 7] = np.asarray(
            self.sim_cfg["box_initial_quaternion_wxyz"], dtype=np.float64
        )
        self.mujoco.mj_forward(self.model, self.data)
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
        self.reset_requested = False

    def _poll_command(self) -> None:
        command = (
            self.command_receiver.poll_latest()
            if self.command_receiver is not None
            else self.unitree_bridge.poll_command()
        )
        if command is not None and command.is_finite():
            self.last_command = command
            self.has_received_command = True
            if command.armed:
                self.physics_started = True
        age_ns = time.monotonic_ns() - self.last_command.timestamp_ns
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

    def _publish(self):
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
        if self.physics_started:
            torque = self._apply_command()
            self.mujoco.mj_step(self.model, self.data)
        else:
            torque = np.zeros(29, dtype=np.float64)
            self.data.ctrl[:] = 0.0
            self.data.time += float(self.model.opt.timestep)
            self.mujoco.mj_forward(self.model, self.data)
        robot_state, task_state = self._publish()
        if not self.recorder.enabled:
            return
        contacts = []
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(
                self.model, self.data, contact_index, force
            )
            contacts.append(
                {
                    "geom1": self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom1),
                    )
                    or int(contact.geom1),
                    "geom2": self.mujoco.mj_id2name(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom2),
                    )
                    or int(contact.geom2),
                    "force_contact_frame": force,
                }
            )
        self.recorder.write(
            {
                "kind": "mujoco_step",
                "sim_time": self.data.time,
                "sequence": robot_state.sequence,
                "joint_pos": robot_state.joint_pos,
                "joint_vel": robot_state.joint_vel,
                "torso_quat_wxyz": robot_state.torso_quat_wxyz,
                "torso_ang_vel": robot_state.torso_ang_vel,
                "raw_action": self.last_command.raw_action,
                "q_target": self.last_command.q_target,
                "torque": torque,
                "box_pos_torso": task_state.box_pos_torso,
                "box_quat_torso_wxyz": task_state.box_quat_torso_wxyz,
                "goal_pos_torso": task_state.goal_pos_torso,
                "success": task_state.success,
                "contact_count": int(self.data.ncon),
                "contacts": contacts,
            }
        )

    def _key_callback(self, keycode: int) -> None:
        if keycode == 259:  # GLFW_BACKSPACE
            self.reset_requested = True
        elif keycode in (81, 256):  # Q or ESC
            self.stop_requested = True

    def run(self, *, duration: float | None = None) -> None:
        period = float(self.sim_cfg["physics_dt"])
        next_tick = time.perf_counter()
        start_time = next_tick
        if self.viewer_enabled:
            viewer_context = self.mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
        else:
            viewer_context = None
        try:
            while not self.stop_requested:
                if duration is not None and time.perf_counter() - start_time >= duration:
                    break
                if viewer_context is not None and not viewer_context.is_running():
                    break
                if self.reset_requested:
                    self.reset()
                self.step()
                if viewer_context is not None:
                    viewer_context.sync()
                if bool(self.sim_cfg.get("realtime", True)):
                    next_tick += period
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
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
    parser.add_argument("--log", help="optional per-physics-step JSONL log")
    parser.add_argument("--duration", type=float, help="optional wall-clock duration")
    args = parser.parse_args()
    cfg = load_deploy_config(args.config)
    server = MujocoServer(
        cfg,
        transport=args.transport or cfg.section("simulation")["transport"],
        viewer=bool(cfg.section("simulation")["viewer"]) and not args.headless,
        log_path=args.log,
    )
    server.run(duration=args.duration)


if __name__ == "__main__":
    main()
