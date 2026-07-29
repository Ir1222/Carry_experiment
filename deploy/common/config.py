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
        return self.checkpoint_path_for()

    @property
    def onnx_path(self) -> Path:
        return self.onnx_path_for()

    @property
    def manifest_path(self) -> Path:
        return self.manifest_path_for()

    @property
    def default_policy_profile(self) -> str:
        policy = self.section("policy")
        return str(policy.get("default_profile", "default"))

    def policy_profile(self, name: str | None = None) -> dict[str, Any]:
        policy = self.section("policy")
        profile_name = self.default_policy_profile if name is None else str(name)
        profiles = policy.get("profiles")
        if isinstance(profiles, dict):
            value = profiles.get(profile_name)
            if not isinstance(value, dict):
                available = ", ".join(sorted(str(item) for item in profiles))
                raise KeyError(
                    f"unknown policy profile {profile_name!r}; "
                    f"available: {available}"
                )
            return value
        if profile_name not in ("default", self.default_policy_profile):
            raise KeyError(f"configuration has no policy profile {profile_name!r}")
        return policy

    def checkpoint_path_for(self, profile: str | None = None) -> Path:
        return self.resolve_path(self.policy_profile(profile)["checkpoint"])

    def onnx_path_for(self, profile: str | None = None) -> Path:
        return self.resolve_path(self.policy_profile(profile)["onnx_path"])

    def manifest_path_for(self, profile: str | None = None) -> Path:
        return self.resolve_path(self.policy_profile(profile)["manifest_path"])

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
    policy_frame = str(robot.get("policy_frame", "")).lower()
    if policy_frame != "pelvis":
        raise ValueError(
            "CarryBox actor was trained with robot.policy_frame='pelvis'"
        )
    if str(robot.get("imu_frame", "")).lower() not in ("torso", "pelvis"):
        raise ValueError("robot.imu_frame must be 'torso' or 'pelvis'")

    profiles = policy.get("profiles")
    if isinstance(profiles, dict):
        if not profiles:
            raise ValueError("policy.profiles must not be empty")
        default_profile = str(policy.get("default_profile", ""))
        if default_profile not in profiles:
            raise ValueError(
                "policy.default_profile must name an entry in policy.profiles"
            )
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(
                    f"policy.profiles.{profile_name} must be a mapping"
                )
            missing = [
                key
                for key in ("checkpoint", "onnx_path", "manifest_path")
                if key not in profile
            ]
            if missing:
                raise ValueError(
                    f"policy profile {profile_name!r} is missing {missing}"
                )

    contact_margin = float(simulation.get("contact_margin", 0.0))
    if contact_margin < 0.0:
        raise ValueError("simulation.contact_margin must be non-negative")
    solref = simulation.get("joint_limit_solref", (0.005, 1.0))
    if (
        not isinstance(solref, (list, tuple))
        or len(solref) != 2
        or float(solref[0]) <= 0.0
        or float(solref[1]) <= 0.0
    ):
        raise ValueError(
            "simulation.joint_limit_solref must be [timeconst, dampratio] "
            "with positive values"
        )
    if float(solref[0]) < 2.0 * physics_dt and not bool(
        simulation.get("disable_refsafe_for_joint_limits", False)
    ):
        raise ValueError(
            "joint-limit timeconst below 2*physics_dt requires "
            "simulation.disable_refsafe_for_joint_limits=true; otherwise "
            "MuJoCo silently clamps it"
        )
    boundary_timeout_ms = float(
        simulation.get("policy_boundary_timeout_ms", 40.0)
    )
    if boundary_timeout_ms <= 0.0:
        raise ValueError(
            "simulation.policy_boundary_timeout_ms must be positive"
        )
    solimp = simulation.get(
        "joint_limit_solimp", (0.9, 0.95, 0.001, 0.5, 2.0)
    )
    if not isinstance(solimp, (list, tuple)) or len(solimp) != 5:
        raise ValueError(
            "simulation.joint_limit_solimp must contain five values"
        )
