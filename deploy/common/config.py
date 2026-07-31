"""YAML configuration loader for deployment executables."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    camera = cfg.section("camera")
    randomization = cfg.section("sim2sim_randomization")

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

    source_platform = simulation.get("source_platform")
    if not isinstance(source_platform, dict):
        raise ValueError("simulation.source_platform must be a mapping")
    if not isinstance(source_platform.get("enabled"), bool):
        raise ValueError("simulation.source_platform.enabled must be boolean")
    platform_size = source_platform.get("size")
    if (
        not isinstance(platform_size, (list, tuple))
        or len(platform_size) != 3
        or not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in platform_size
        )
    ):
        raise ValueError(
            "simulation.source_platform.size must contain three positive "
            "finite values"
        )
    platform_gap = float(source_platform.get("box_gap", -1.0))
    if not math.isfinite(platform_gap) or platform_gap < 0.0:
        raise ValueError(
            "simulation.source_platform.box_gap must be finite and "
            "non-negative"
        )
    platform_friction = source_platform.get("friction")
    if (
        not isinstance(platform_friction, (list, tuple))
        or len(platform_friction) != 3
        or not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in platform_friction
        )
    ):
        raise ValueError(
            "simulation.source_platform.friction must contain three "
            "non-negative finite values"
        )

    if str(camera.get("name", "")) != "d455_camera":
        raise ValueError("camera.name must be 'd455_camera'")
    if str(camera.get("body", "")) != "d455_link":
        raise ValueError("CarryBox camera.body must be 'd455_link'")
    position = camera.get("position")
    quaternion = camera.get("quaternion_wxyz")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("camera.position must contain three values")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("camera.quaternion_wxyz must contain four values")
    if not all(math.isfinite(float(value)) for value in position):
        raise ValueError("camera.position must be finite")
    quaternion_norm = math.sqrt(
        sum(float(value) ** 2 for value in quaternion)
    )
    if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-6):
        raise ValueError(
            "camera.quaternion_wxyz must have unit norm, got "
            f"{quaternion_norm:.9g}"
        )
    width = int(camera.get("width", 0))
    height = int(camera.get("height", 0))
    vertical_fov_deg = float(camera.get("vertical_fov_deg", 0.0))
    near_m = float(camera.get("near_m", 0.0))
    far_m = float(camera.get("far_m", 0.0))
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("camera.vertical_fov_deg must be in (0, 180)")
    if near_m <= 0.0 or far_m <= near_m:
        raise ValueError("camera range must satisfy 0 < near_m < far_m")
    horizontal_fov_deg = math.degrees(
        2.0
        * math.atan(
            (width / height)
            * math.tan(math.radians(vertical_fov_deg) / 2.0)
        )
    )
    if not 85.0 <= horizontal_fov_deg <= 90.0:
        raise ValueError(
            "camera resolution/fovy must reproduce the trained D455 "
            f"85-90 degree horizontal FOV, got {horizontal_fov_deg:.3f}"
        )

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
    collision = simulation.get("collision")
    if not isinstance(collision, dict):
        raise ValueError("simulation.collision must be a mapping")
    collision_profiles = collision.get("profiles")
    expected_collision_profiles = {
        "current",
        "no_robot_self",
        "isaac_parity",
    }
    if (
        not isinstance(collision_profiles, dict)
        or set(collision_profiles) != expected_collision_profiles
    ):
        raise ValueError(
            "simulation.collision.profiles must define current, "
            "no_robot_self, and isaac_parity"
        )
    default_collision_profile = str(
        collision.get("default_profile", "current")
    )
    if default_collision_profile not in collision_profiles:
        raise ValueError(
            "simulation.collision.default_profile must name a profile"
        )
    for name, collision_profile in collision_profiles.items():
        if not isinstance(collision_profile, dict):
            raise ValueError(
                f"simulation.collision.profiles.{name} must be a mapping"
            )
        if not isinstance(
            collision_profile.get("disable_robot_self"), bool
        ):
            raise ValueError(
                f"collision profile {name} disable_robot_self must be boolean"
            )
        for margin_name in ("robot_margin", "external_margin"):
            margin = float(collision_profile.get(margin_name, -1.0))
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(
                    f"collision profile {name} {margin_name} must be "
                    "finite and non-negative"
                )
        exclusions = collision_profile.get("exclude_body_pairs")
        if not isinstance(exclusions, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(body, str) and body for body in pair)
            for pair in exclusions
        ):
            raise ValueError(
                f"collision profile {name} exclude_body_pairs must be a "
                "list of two-body-name lists"
            )
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

    profiles = randomization.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "nominal",
        "light",
        "train_match",
    }:
        raise ValueError(
            "sim2sim_randomization.profiles must define nominal, light, "
            "and train_match"
        )
    default_randomization = str(
        randomization.get("default_profile", "nominal")
    )
    if default_randomization not in profiles:
        raise ValueError(
            "sim2sim_randomization.default_profile must name a profile"
        )
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(
                f"sim2sim_randomization.profiles.{name} must be a mapping"
            )
        scale = float(profile.get("scale", -1.0))
        max_delay = int(profile.get("max_action_delay_steps", -1))
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                f"randomization profile {name} scale must be in [0, 1]"
            )
        if not 0 <= max_delay <= 4:
            raise ValueError(
                f"randomization profile {name} delay must be in [0, 4]"
            )
    ranges = randomization.get("train_match_ranges")
    if not isinstance(ranges, dict):
        raise ValueError(
            "sim2sim_randomization.train_match_ranges must be a mapping"
        )
    required_ranges = (
        "joint_position_offset",
        "root_linear_velocity",
        "root_angular_velocity",
        "box_scale_x",
        "box_scale_y",
        "box_scale_z",
        "box_density",
        "box_xy_abs",
        "box_z",
        "goal_distance",
        "goal_bearing_abs_deg",
        "goal_z",
        "robot_friction",
        "link_mass_scale",
        "torso_payload_mass",
        "torso_com_displacement",
        "kp_factor",
        "kd_factor",
        "motor_strength",
        "torque_bias_fraction",
    )
    for key in required_ranges:
        value = ranges.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(math.isfinite(float(item)) for item in value)
            or float(value[0]) > float(value[1])
        ):
            raise ValueError(
                f"sim2sim_randomization.train_match_ranges.{key} "
                "must be a finite [low, high] range"
            )
    if float(ranges["box_density"][0]) <= 0.0:
        raise ValueError("randomized box density must remain positive")
    if float(ranges["robot_friction"][0]) <= 0.0:
        raise ValueError("randomized friction must remain positive")
    if float(ranges["link_mass_scale"][0]) <= 0.0:
        raise ValueError("randomized link mass scale must remain positive")
    for key in ("kp_factor", "kd_factor", "motor_strength"):
        if float(ranges[key][0]) <= 0.0:
            raise ValueError(f"randomized {key} must remain positive")
    noise = ranges.get("observation_noise")
    required_noise = (
        "gyro",
        "gravity_rotation",
        "joint_position",
        "joint_velocity",
        "endpoint",
        "task_position",
        "task_orientation_deg",
    )
    if not isinstance(noise, dict) or any(
        key not in noise
        or not math.isfinite(float(noise[key]))
        or float(noise[key]) < 0.0
        for key in required_noise
    ):
        raise ValueError(
            "sim2sim randomization observation noise values must be "
            "finite and non-negative"
        )
    interval = int(ranges.get("disturbance_interval_policy_steps", 0))
    force = float(ranges.get("disturbance_force", -1.0))
    if interval <= 0 or not math.isfinite(force) or force < 0.0:
        raise ValueError(
            "disturbance interval must be positive and force non-negative"
        )
