"""YAML configuration loader for deployment executables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    END_EFFECTOR_NAMES,
    FRAME_OBS_DIM,
    HISTORY_LENGTH,
)
from .mapping import validate_motor_mapping


@dataclass(frozen=True, slots=True)
class DeployConfig:
    path: Path
    project_root: Path
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"missing config section: {name}")
        return value

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self.resolve_path(self.section("policy")["checkpoint"])

    @property
    def onnx_path(self) -> Path:
        return self.resolve_path(self.section("policy")["onnx_path"])

    @property
    def urdf_path(self) -> Path:
        return self.resolve_path(self.section("robot")["urdf"])

    @property
    def mjcf_path(self) -> Path:
        return self.resolve_path(self.section("simulation")["mjcf"])


def load_deploy_config(path: str | Path) -> DeployConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")

    root_value = data.get("project_root", "../..")
    project_root = (config_path.parent / root_value).resolve()
    cfg = DeployConfig(path=config_path, project_root=project_root, data=data)
    _validate(cfg)
    return cfg


def _validate(cfg: DeployConfig) -> None:
    policy = cfg.section("policy")
    control = cfg.section("control")
    simulation = cfg.section("simulation")
    robot = cfg.section("robot")

    expected = {
        "actor_obs_dim": ACTOR_OBS_DIM,
        "frame_obs_dim": FRAME_OBS_DIM,
        "history_length": HISTORY_LENGTH,
        "action_dim": ACTION_DIM,
    }
    for key, expected_value in expected.items():
        actual = int(policy.get(key, expected_value))
        if actual != expected_value:
            raise ValueError(f"policy.{key} must be {expected_value}, got {actual}")

    physics_dt = float(simulation["physics_dt"])
    policy_hz = float(control["policy_hz"])
    decimation = int(control["decimation"])
    if physics_dt <= 0.0 or policy_hz <= 0.0 or decimation <= 0:
        raise ValueError("physics_dt, policy_hz, and decimation must be positive")
    expected_policy_dt = physics_dt * decimation
    if abs(expected_policy_dt - 1.0 / policy_hz) > 1e-9:
        raise ValueError(
            "inconsistent control timing: physics_dt * decimation "
            f"= {expected_policy_dt}, but policy_hz implies {1.0 / policy_hz}"
        )
    physics_hz = float(control["physics_hz"])
    if abs(physics_hz - 1.0 / physics_dt) > 1e-9:
        raise ValueError(
            f"control.physics_hz={physics_hz} does not match "
            f"simulation.physics_dt={physics_dt}"
        )
    endpoints = tuple(robot.get("end_effectors", ()))
    if endpoints != END_EFFECTOR_NAMES:
        raise ValueError(
            "robot.end_effectors must preserve the actor observation order: "
            f"{END_EFFECTOR_NAMES}"
        )
    validate_motor_mapping(robot["policy_to_motor"])
    if str(robot.get("imu_frame", "")).lower() not in ("torso", "pelvis"):
        raise ValueError("robot.imu_frame must be 'torso' or 'pelvis'")
