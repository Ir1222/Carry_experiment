"""Run the shared CarryBox policy against MuJoCo or a real Unitree G1 backend."""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from deploy.common.config import load_deploy_config
from deploy.common.constants import DEFAULT_DOF_POS, KD, KP
from deploy.common.control import PDController
from deploy.common.kinematics import MujocoKinematicsProvider
from deploy.common.jsonl import JsonlRecorder
from deploy.common.mapping import RobotDescription
from deploy.common.observation import ObservationBuilder
from deploy.common.safety import SafetyGate
from deploy.policy.backends import UdpPolicyBackend
from deploy.policy.core import PolicyCore
from deploy.policy.inference import OnnxActor
from deploy.tools.build_mjcf import build_robot_mjcf


class OperatorState:
    def __init__(self, *, armed: bool) -> None:
        self.armed = armed
        self.quit = False
        self._lock = threading.Lock()

    def set_armed(self, value: bool) -> None:
        with self._lock:
            self.armed = bool(value)

    def snapshot(self) -> tuple[bool, bool]:
        with self._lock:
            return self.armed, self.quit

    def request_quit(self) -> None:
        with self._lock:
            self.quit = True


def _console_loop(operator: OperatorState, safety: SafetyGate) -> None:
    print(
        "Controls: ']' or 'arm' = arm, 'o' or 'hold' = hold, "
        "'x' or 'estop' = latched e-stop, 'clear' = clear e-stop, 'q' = quit"
    )
    while True:
        try:
            command = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if command in ("]", "arm"):
            safety.clear_estop()
            operator.set_armed(True)
            print("Policy arm requested")
        elif command in ("o", "hold"):
            operator.set_armed(False)
            print("Current-position damping hold requested")
        elif command in ("x", "estop"):
            safety.trigger_estop()
            operator.set_armed(False)
            print("Emergency stop latched")
        elif command == "clear":
            safety.clear_estop()
            operator.set_armed(False)
            print("Emergency stop cleared; policy remains disarmed")
        elif command in ("q", "quit", "exit"):
            operator.request_quit()
            return


def _address(value) -> tuple[str, int]:
    return str(value[0]), int(value[1])


def _make_backend(cfg, *, mode: str, transport: str, write_enabled: bool):
    network = cfg.section("network")
    if mode == "sim2sim" and transport == "udp":
        return UdpPolicyBackend(
            _address(network["robot_state_udp"]),
            _address(network["task_state_udp"]),
            _address(network["robot_command_udp"]),
        )

    generated = cfg.resolve_path(cfg.section("simulation")["generated_robot_mjcf"])
    if not generated.exists():
        build_robot_mjcf(
            cfg.urdf_path,
            generated,
            joint_armature=float(
                cfg.section("simulation").get("joint_armature", 0.01)
            ),
        )
    kinematics = MujocoKinematicsProvider(
        generated,
        RobotDescription.from_urdf(cfg.urdf_path),
        pelvis_body=cfg.section("robot")["base_body"],
        torso_body=cfg.section("robot")["torso_body"],
        end_effector_names=tuple(cfg.section("robot")["end_effectors"]),
    )
    from deploy.sim2real.unitree_backend import UnitreePolicyBackend

    command_hz = float(cfg.section("control")["hardware_command_hz"])
    return UnitreePolicyBackend(
        domain_id=int(network["domain_id"]),
        interface=str(network["interface"]),
        policy_to_motor=cfg.section("robot")["policy_to_motor"],
        kinematics=kinematics,
        task_address=_address(network["task_state_udp"]),
        command_hz=command_hz,
        write_enabled=write_enabled,
        imu_frame=str(cfg.section("robot")["imu_frame"]),
        command_stale_timeout_ms=float(
            cfg.section("safety")["command_stale_timeout_ms"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--mode", choices=("sim2sim", "sim2real"), default="sim2sim")
    parser.add_argument("--transport", choices=("unitree_dds", "udp"))
    parser.add_argument("--model", help="override ONNX model")
    parser.add_argument("--arm", action="store_true", help="start armed (sim recommended)")
    parser.add_argument(
        "--allow-hardware-command",
        action="store_true",
        help="second real-hardware interlock; config safety.dry_run must also be false",
    )
    parser.add_argument("--duration", type=float, help="optional run duration in seconds")
    parser.add_argument("--log", help="optional per-policy-step JSONL log")
    args = parser.parse_args()

    cfg = load_deploy_config(args.config)
    transport = args.transport or cfg.section("simulation")["transport"]
    model_path = cfg.resolve_path(args.model) if args.model else cfg.onnx_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX actor not found: {model_path}\n"
            "Run: python -m deploy.tools.export_actor "
            "--checkpoint legged_gym/logs/Jul09_from_55500/model_73500.pt"
        )

    safety_cfg = cfg.section("safety")
    configured_dry_run = bool(safety_cfg["dry_run"])
    if args.mode == "sim2real":
        write_enabled = args.allow_hardware_command and not configured_dry_run
        if args.allow_hardware_command and configured_dry_run:
            raise RuntimeError(
                "--allow-hardware-command was given, but safety.dry_run is still true"
            )
        dry_run = configured_dry_run
    else:
        write_enabled = True
        dry_run = False

    robot = RobotDescription.from_urdf(cfg.urdf_path)
    policy_cfg = cfg.section("policy")
    control_cfg = cfg.section("control")
    builder = ObservationBuilder(
        DEFAULT_DOF_POS,
        clip=float(policy_cfg["clip_observations"]),
        legacy_ankle_delay_steps=int(policy_cfg["legacy_ankle_delay_steps"]),
    )
    controller = PDController(
        robot,
        default_dof_pos=DEFAULT_DOF_POS,
        kp=KP,
        kd=KD,
        action_scale=float(control_cfg["action_scale"]),
        action_clip=float(policy_cfg["clip_actions"]),
    )
    safety = SafetyGate(
        robot,
        max_robot_state_age_ms=float(safety_cfg["max_robot_state_age_ms"]),
        max_task_state_age_ms=float(safety_cfg["max_task_state_age_ms"]),
        max_projected_gravity_xy=float(
            safety_cfg["max_projected_gravity_xy"]
        ),
        joint_limit_margin=float(safety_cfg["joint_limit_margin"]),
    )
    core = PolicyCore(
        OnnxActor(model_path),
        builder,
        controller,
        max_inference_time_ms=float(safety_cfg["max_inference_time_ms"]),
    )
    backend = _make_backend(
        cfg,
        mode=args.mode,
        transport=transport,
        write_enabled=write_enabled,
    )
    recorder = JsonlRecorder(args.log)
    operator = OperatorState(armed=args.arm)
    if args.mode == "sim2real" and args.arm and not write_enabled:
        print("Real backend is armed for inference only; hardware writes are disabled")
    if sys.stdin.isatty():
        threading.Thread(
            target=_console_loop, args=(operator, safety), daemon=True
        ).start()

    policy_period = 1.0 / float(control_cfg["policy_hz"])
    next_tick = time.perf_counter()
    start_time = next_tick
    last_armed = False
    armed_since: float | None = None
    last_log = start_time
    last_robot_sequence = -1
    try:
        while True:
            armed, quit_requested = operator.snapshot()
            if quit_requested:
                break
            if args.duration is not None and time.perf_counter() - start_time >= args.duration:
                break
            robot_state, task_state = backend.poll()
            if robot_state is not None and task_state is not None:
                state_sequence_reset = (
                    last_robot_sequence >= 0
                    and robot_state.sequence <= last_robot_sequence
                )
                if (armed and not last_armed) or state_sequence_reset:
                    core.reset(robot_state)
                    armed_since = time.perf_counter()
                elif not armed:
                    armed_since = None
                decision = safety.evaluate(
                    robot_state,
                    task_state,
                    armed=armed,
                    dry_run=dry_run,
                )
                if (
                    armed
                    and not decision.allowed
                    and decision.reason != "dry-run blocks command output"
                ):
                    operator.set_armed(False)
                    armed = False
                if (
                    robot_state.sequence != last_robot_sequence
                    or state_sequence_reset
                ):
                    gain_ramp = 1.0
                    if args.mode == "sim2real":
                        ramp_seconds = float(safety_cfg["gain_ramp_seconds"])
                        elapsed = (
                            0.0
                            if armed_since is None
                            else time.perf_counter() - armed_since
                        )
                        gain_ramp = (
                            1.0
                            if ramp_seconds <= 0.0
                            else float(np.clip(elapsed / ramp_seconds, 0.0, 1.0))
                        )
                    step = core.step(
                        robot_state,
                        task_state,
                        command_allowed=decision.allowed,
                        reason=decision.reason,
                        hardware_safe=args.mode == "sim2real",
                        update_action_history=armed,
                        run_inference_when_blocked=(
                            decision.reason == "dry-run blocks command output"
                        ),
                        kp_scale=(
                            float(control_cfg["hardware_kp_scale"])
                            * gain_ramp
                            if args.mode == "sim2real"
                            else 1.0
                        ),
                        kd_scale=(
                            float(control_cfg["hardware_kd_scale"])
                            if args.mode == "sim2real"
                            else 1.0
                        ),
                    )
                    if armed and decision.allowed and not step.command.armed:
                        operator.set_armed(False)
                        armed = False
                    backend.send(step.command)
                    recorder.write(
                        {
                            "kind": "policy_step",
                            "sequence": robot_state.sequence,
                            "timestamp_ns": robot_state.timestamp_ns,
                            "observation_slices": builder.frame_slices(
                                step.actor_obs[0, -123:]
                            ),
                            "raw_action": step.raw_action,
                            "q_target": step.command.q_target,
                            "kp": step.command.kp,
                            "kd": step.command.kd,
                            "command_armed": step.command.armed,
                            "command_reason": step.command.reason,
                            "inference_time_ms": step.inference_time_ms,
                        }
                    )
                    last_robot_sequence = robot_state.sequence
                    now = time.perf_counter()
                    if now - last_log >= 1.0:
                        print(
                            f"mode={args.mode} transport={transport} "
                            f"armed={armed} allowed={decision.allowed} "
                            f"reason={decision.reason} "
                            f"action_norm={np.linalg.norm(step.raw_action):.3f} "
                            f"inference_ms={step.inference_time_ms:.3f} "
                            f"gain_ramp={gain_ramp:.3f}"
                        )
                        last_log = now
            last_armed = armed
            next_tick += policy_period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        operator.set_armed(False)
        backend.close()
        recorder.close()


if __name__ == "__main__":
    main()
