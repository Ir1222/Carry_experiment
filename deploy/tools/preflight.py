"""Deployment preflight checks for PhysHSI CarryBox.

The checks in this module are intentionally read-only.  In particular, the
sim2real dry-run preflight imports the Unitree message types but never creates
a publisher and never writes a LowCmd.
"""

from __future__ import annotations

import argparse
import importlib
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from deploy.common.config import DeployConfig, load_deploy_config
from deploy.common.constants import ACTION_DIM, ACTOR_OBS_DIM
from deploy.common.mapping import RobotDescription
from deploy.common.model_manifest import validate_model_manifest
from deploy.policy.inference import OnnxActor


PREFLIGHT_MODES = ("udp-sim2sim", "dds-sim2sim", "sim2real-dryrun")


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, callback: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, passed=True, detail=callback())
    except Exception as exc:  # A preflight must report every failure together.
        return CheckResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")


def _interface_names() -> set[str]:
    return {name for _, name in socket.if_nameindex()}


def _require_interface(name: str, *, allow_loopback: bool) -> str:
    names = _interface_names()
    if name not in names:
        available = ", ".join(sorted(names)) or "<none>"
        raise RuntimeError(f"network interface {name!r} does not exist; available: {available}")
    if not allow_loopback and name in {"lo", "lo0"}:
        raise RuntimeError("sim2real requires the dedicated G1 Ethernet interface, not loopback")
    return f"interface={name}"


def _check_python() -> str:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10+ is required by this deployment package; got {sys.version.split()[0]}"
        )
    return sys.version.split()[0]


def _check_imports() -> str:
    versions: list[str] = []
    for module_name in ("torch", "mujoco", "onnxruntime"):
        module = importlib.import_module(module_name)
        versions.append(f"{module_name}={getattr(module, '__version__', 'unknown')}")
    return ", ".join(versions)


def _check_required_files(cfg: DeployConfig, profile: str) -> str:
    simulation = cfg.section("simulation")
    required = {
        "URDF": cfg.urdf_path,
        "scene MJCF": cfg.mjcf_path,
        "generated MJCF": cfg.resolve_path(simulation["generated_robot_mjcf"]),
        "ONNX actor": cfg.onnx_path_for(profile),
        "manifest": cfg.manifest_path_for(profile),
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required deployment files: " + "; ".join(missing))
    return ", ".join(f"{label}={path.name}" for label, path in required.items())


def _check_manifest(cfg: DeployConfig, profile: str) -> str:
    control = cfg.section("control")
    manifest = validate_model_manifest(
        cfg.manifest_path_for(profile),
        cfg.onnx_path_for(profile),
        profile=profile,
        policy_frame=str(cfg.section("robot")["policy_frame"]),
        action_scale=float(control["action_scale"]),
        physics_hz=int(control["physics_hz"]),
        policy_hz=int(control["policy_hz"]),
        checkpoint_path=cfg.checkpoint_path_for(profile),
    )
    return (
        f"profile={profile}, actor={ACTOR_OBS_DIM}->{ACTION_DIM}, "
        f"sha256={manifest['onnx_sha256']}"
    )


def _check_actor(cfg: DeployConfig, profile: str) -> str:
    actor = OnnxActor(cfg.onnx_path_for(profile))
    output = actor(np.zeros((1, ACTOR_OBS_DIM), dtype=np.float32))
    if output.shape != (1, ACTION_DIM):
        raise RuntimeError(f"unexpected actor output shape {output.shape}")
    if not np.all(np.isfinite(output)):
        raise RuntimeError("actor produced NaN/Inf for a zero observation")
    return (
        f"actor_obs[1,{ACTOR_OBS_DIM}] -> actions[1,{ACTION_DIM}]"
    )


def _check_robot_description(cfg: DeployConfig) -> str:
    endpoints = tuple(cfg.section("robot")["end_effectors"])
    robot = RobotDescription.from_urdf(cfg.urdf_path)
    if len(robot.joint_names) != ACTION_DIM:
        raise RuntimeError(f"URDF mapping contains {len(robot.joint_names)} joints")
    if np.any(robot.effort_limits <= 0.0):
        raise RuntimeError("URDF contains non-positive effort limits")
    return f"{len(robot.joint_names)} joints, {len(endpoints)} endpoints"


def _check_scene(cfg: DeployConfig) -> str:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
    if model.nu != ACTION_DIM:
        raise RuntimeError(f"MuJoCo scene has {model.nu} actuators, expected {ACTION_DIM}")
    physics_dt = float(cfg.section("simulation")["physics_dt"])
    if not np.isclose(float(model.opt.timestep), physics_dt):
        raise RuntimeError(
            f"MuJoCo timestep={model.opt.timestep}, config physics_dt={physics_dt}"
        )
    return f"nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep:g}"


def _check_frequencies(cfg: DeployConfig) -> str:
    control = cfg.section("control")
    physics_hz = float(control["physics_hz"])
    policy_hz = float(control["policy_hz"])
    command_hz = float(control["hardware_command_hz"])
    if not np.isclose(physics_hz, 200.0):
        raise RuntimeError(f"physics frequency must be 200 Hz, got {physics_hz:g}")
    if not np.isclose(policy_hz, 50.0):
        raise RuntimeError(f"policy frequency must be 50 Hz, got {policy_hz:g}")
    if not np.isclose(command_hz, 500.0):
        raise RuntimeError(f"hardware command frequency must be 500 Hz, got {command_hz:g}")
    return f"physics={physics_hz:g}Hz, policy={policy_hz:g}Hz, command={command_hz:g}Hz"


def _check_safety_thresholds(cfg: DeployConfig) -> str:
    safety = cfg.section("safety")
    inference_ms = float(safety["max_inference_time_ms"])
    robot_ms = float(safety["max_robot_state_age_ms"])
    task_ms = float(safety["max_task_state_age_ms"])
    command_ms = float(safety["command_stale_timeout_ms"])
    values = (inference_ms, robot_ms, task_ms, command_ms)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("all inference/state/command timeout thresholds must be finite and positive")
    if inference_ms > 15.0:
        raise RuntimeError(f"max inference time must be <=15 ms, got {inference_ms:g}")
    if robot_ms > 40.0:
        raise RuntimeError(f"max robot-state age must be <=40 ms, got {robot_ms:g}")
    if task_ms > 100.0:
        raise RuntimeError(f"max task-state age must be <=100 ms, got {task_ms:g}")
    if command_ms > 200.0:
        raise RuntimeError(f"command stale timeout must be <=200 ms, got {command_ms:g}")
    return (
        f"inference={inference_ms:g}ms, robot={robot_ms:g}ms, "
        f"task={task_ms:g}ms, command={command_ms:g}ms"
    )


def _check_udp_ports(cfg: DeployConfig) -> str:
    network = cfg.section("network")
    addresses = (
        tuple(network["robot_state_udp"]),
        tuple(network["task_state_udp"]),
        tuple(network["robot_command_udp"]),
    )
    sockets: list[socket.socket] = []
    try:
        for host, port in addresses:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((host, port))
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return ", ".join(f"{host}:{port}" for host, port in addresses)


def _check_unitree_sdk() -> str:
    channel = importlib.import_module("unitree_sdk2py.core.channel")
    messages = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    for symbol, module in (
        ("ChannelFactoryInitialize", channel),
        ("LowState_", messages),
        ("LowCmd_", messages),
    ):
        if not hasattr(module, symbol):
            raise ImportError(f"unitree_sdk2py is missing {symbol}")
    return "ChannelFactoryInitialize, LowState_, LowCmd_ import successfully"


def _check_dryrun_lock(cfg: DeployConfig) -> str:
    dry_run = bool(cfg.section("safety")["dry_run"])
    hardware_kp_scale = float(cfg.section("control")["hardware_kp_scale"])
    if not dry_run:
        raise RuntimeError("safety.dry_run must be true")
    if not np.isclose(hardware_kp_scale, 0.0):
        raise RuntimeError("control.hardware_kp_scale must be 0.0")
    return "dry_run=true, hardware_kp_scale=0.0, LowCmd writes disabled"


def run_preflight(
    config_path: str | Path,
    mode: str,
    *,
    check_ports: bool = True,
    interface_override: str | None = None,
    profile: str | None = None,
) -> list[CheckResult]:
    """Run read-only deployment checks and return all results."""

    if mode not in PREFLIGHT_MODES:
        raise ValueError(f"unknown preflight mode {mode!r}")
    cfg = load_deploy_config(config_path)
    selected_profile = profile or cfg.default_policy_profile
    cfg.policy_profile(selected_profile)
    interface = interface_override or str(cfg.section("network")["interface"])

    checks: list[tuple[str, Callable[[], str]]] = [
        ("Python", _check_python),
        ("dependencies", _check_imports),
        (
            "deployment files",
            lambda: _check_required_files(cfg, selected_profile),
        ),
        ("manifest", lambda: _check_manifest(cfg, selected_profile)),
        ("ONNX actor", lambda: _check_actor(cfg, selected_profile)),
        ("joint/body mapping", lambda: _check_robot_description(cfg)),
        ("MuJoCo scene", lambda: _check_scene(cfg)),
        ("control rates", lambda: _check_frequencies(cfg)),
        ("safety timeouts", lambda: _check_safety_thresholds(cfg)),
    ]
    if mode == "udp-sim2sim" and check_ports:
        checks.append(("UDP ports", lambda: _check_udp_ports(cfg)))
    if mode in {"dds-sim2sim", "sim2real-dryrun"}:
        checks.extend(
            [
                ("Unitree SDK2", _check_unitree_sdk),
                (
                    "DDS interface",
                    lambda: _require_interface(
                        interface, allow_loopback=mode == "dds-sim2sim"
                    ),
                ),
            ]
        )
    if mode == "sim2real-dryrun":
        checks.append(("hardware safety lock", lambda: _check_dryrun_lock(cfg)))

    return [_check(name, callback) for name, callback in checks]


def _print_results(results: Iterable[CheckResult]) -> bool:
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        all_passed &= result.passed
    return all_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--mode", choices=PREFLIGHT_MODES, default="udp-sim2sim")
    parser.add_argument("--profile", help="named policy profile")
    parser.add_argument(
        "--interface",
        help="override network.interface for this read-only check",
    )
    parser.add_argument(
        "--skip-port-check",
        action="store_true",
        help="do not verify that the three UDP ports are currently free",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_preflight(
        args.config,
        args.mode,
        check_ports=not args.skip_port_check,
        interface_override=args.interface,
        profile=args.profile,
    )
    passed = _print_results(results)
    print(
        f"Preflight {'passed' if passed else 'failed'}: "
        f"{sum(result.passed for result in results)}/{len(results)} checks passed."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
