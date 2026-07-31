"""Seeded, serializable randomization scenarios for MuJoCo Sim2Sim."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from deploy.common.constants import ACTION_DIM, DEFAULT_DOF_POS
from deploy.common.types import RobotState, TaskState


PROFILE_NAMES = ("nominal", "light", "train_match")
_CATEGORY_IDS = {
    "reset": 101,
    "physics": 211,
    "actuator": 307,
    "sensor": 401,
    "disturbance": 503,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_quat_wxyz(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("quaternion must have a positive finite norm")
    return quat / norm


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def _axis_angle_quat(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    axis = vector / angle
    half = 0.5 * angle
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )


def _vector(value: Any, size: int, name: str) -> list[float]:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result.tolist()


@dataclass(frozen=True, slots=True)
class ScenarioSample:
    """A complete episode definition shared by every evaluated policy."""

    format_version: int
    profile: str
    seed: int
    episode_index: int
    approximation: str
    robot_position: list[float]
    robot_quaternion_wxyz: list[float]
    root_linear_velocity: list[float]
    root_angular_velocity: list[float]
    joint_position: list[float]
    box_size: list[float]
    box_density: float
    box_position: list[float]
    box_quaternion_wxyz: list[float]
    goal_position: list[float]
    robot_friction: float
    link_mass_scale_seed: int
    link_mass_scale_range: list[float]
    torso_payload_mass: float
    torso_com_displacement: list[float]
    kp_factors: list[float]
    kd_factors: list[float]
    motor_strength: list[float]
    torque_bias_fraction: list[float]
    action_delay_steps: int
    sensor_seed: int
    gyro_noise: float
    gravity_rotation_noise: float
    joint_pos_noise: float
    joint_vel_noise: float
    endpoint_noise: float
    task_position_noise: float
    task_orientation_noise_rad: float
    disturbance_seed: int
    disturbance_interval_policy_steps: int
    disturbance_force: float

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("ScenarioSample.format_version must be 1")
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"unknown randomization profile {self.profile!r}")
        for name in (
            "robot_position",
            "root_linear_velocity",
            "root_angular_velocity",
            "box_size",
            "box_position",
            "goal_position",
            "torso_com_displacement",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        object.__setattr__(
            self,
            "robot_quaternion_wxyz",
            _normalize_quat_wxyz(
                np.asarray(self.robot_quaternion_wxyz, dtype=np.float64)
            ).tolist(),
        )
        object.__setattr__(
            self,
            "box_quaternion_wxyz",
            _normalize_quat_wxyz(
                np.asarray(self.box_quaternion_wxyz, dtype=np.float64)
            ).tolist(),
        )
        for name in (
            "joint_position",
            "kp_factors",
            "kd_factors",
            "motor_strength",
            "torque_bias_fraction",
        ):
            object.__setattr__(
                self, name, _vector(getattr(self, name), ACTION_DIM, name)
            )
        object.__setattr__(
            self,
            "link_mass_scale_range",
            _vector(self.link_mass_scale_range, 2, "link_mass_scale_range"),
        )
        if any(value <= 0.0 for value in self.box_size):
            raise ValueError("box_size values must be positive")
        if self.box_density <= 0.0 or self.robot_friction <= 0.0:
            raise ValueError("box_density and robot_friction must be positive")
        if any(value <= 0.0 for value in self.kp_factors + self.kd_factors):
            raise ValueError("KP/KD factors must be positive")
        if any(value <= 0.0 for value in self.motor_strength):
            raise ValueError("motor strength factors must be positive")
        if not 0 <= self.action_delay_steps <= 4:
            raise ValueError("action_delay_steps must be in [0, 4]")
        if self.disturbance_interval_policy_steps <= 0:
            raise ValueError("disturbance interval must be positive")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @property
    def fingerprint(self) -> str:
        payload = _canonical_json(self.to_dict(include_fingerprint=False)).encode()
        return hashlib.sha256(payload).hexdigest()

    def write(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def read(cls, path: str | Path) -> "ScenarioSample":
        source = Path(path).expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        expected_fingerprint = data.pop("fingerprint", None)
        scenario = cls(**data)
        if (
            expected_fingerprint is not None
            and expected_fingerprint != scenario.fingerprint
        ):
            raise ValueError(f"scenario fingerprint mismatch: {source}")
        return scenario

    def link_mass_factors(self, body_count: int) -> np.ndarray:
        if body_count <= 0:
            raise ValueError("body_count must be positive")
        low, high = self.link_mass_scale_range
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.link_mass_scale_seed, _CATEGORY_IDS["physics"], body_count]
            )
        )
        return rng.uniform(low, high, size=body_count)

    def disturbance_at(self, policy_step: int) -> np.ndarray:
        if (
            self.disturbance_force <= 0.0
            or policy_step <= 0
            or policy_step % self.disturbance_interval_policy_steps != 0
        ):
            return np.zeros(3, dtype=np.float64)
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.disturbance_seed,
                    _CATEGORY_IDS["disturbance"],
                    int(policy_step),
                ]
            )
        )
        return rng.uniform(
            -self.disturbance_force, self.disturbance_force, size=3
        )

    def observed_states(
        self,
        robot: RobotState,
        task: TaskState,
        *,
        sample_index: int | None = None,
    ) -> tuple[RobotState, TaskState]:
        """Return a deterministic noisy sensor view for one source sequence."""

        if self.profile == "nominal":
            return robot, task
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.sensor_seed,
                    _CATEGORY_IDS["sensor"],
                    int(robot.sequence if sample_index is None else sample_index),
                ]
            )
        )
        rotation_noise = rng.uniform(
            -self.gravity_rotation_noise,
            self.gravity_rotation_noise,
            size=3,
        )
        observed_quat = _normalize_quat_wxyz(
            _quat_mul_wxyz(
                np.asarray(robot.policy_frame_quat_wxyz),
                _axis_angle_quat(rotation_noise),
            )
        )
        task_rotation_axis = rng.normal(size=3)
        task_rotation_axis /= max(float(np.linalg.norm(task_rotation_axis)), 1e-12)
        task_rotation_angle = rng.uniform(
            -self.task_orientation_noise_rad,
            self.task_orientation_noise_rad,
        )
        observed_box_quat = _normalize_quat_wxyz(
            _quat_mul_wxyz(
                np.asarray(task.box_quat_policy_frame_wxyz),
                _axis_angle_quat(task_rotation_axis * task_rotation_angle),
            )
        )
        observed_robot = RobotState(
            sequence=robot.sequence,
            timestamp_ns=robot.timestamp_ns,
            policy_frame_quat_wxyz=observed_quat,
            policy_frame_ang_vel=robot.policy_frame_ang_vel
            + rng.uniform(-self.gyro_noise, self.gyro_noise, size=3),
            joint_pos=robot.joint_pos
            + rng.uniform(-self.joint_pos_noise, self.joint_pos_noise, ACTION_DIM),
            joint_vel=robot.joint_vel
            + rng.uniform(-self.joint_vel_noise, self.joint_vel_noise, ACTION_DIM),
            end_effector_pos_policy_frame=robot.end_effector_pos_policy_frame
            + rng.uniform(-self.endpoint_noise, self.endpoint_noise, (5, 3)),
        )
        observed_task = TaskState(
            sequence=task.sequence,
            timestamp_ns=task.timestamp_ns,
            box_pos_policy_frame=task.box_pos_policy_frame
            + rng.uniform(
                -self.task_position_noise, self.task_position_noise, size=3
            ),
            box_quat_policy_frame_wxyz=observed_box_quat,
            box_size=task.box_size.copy(),
            goal_pos_policy_frame=task.goal_pos_policy_frame
            + rng.uniform(
                -self.task_position_noise, self.task_position_noise, size=3
            ),
            success=task.success,
        )
        return observed_robot, observed_task


class ScenarioSampler:
    """Sample stable episode definitions from deployment YAML."""

    def __init__(self, deploy_config) -> None:
        self.cfg = deploy_config
        self.sim_cfg = deploy_config.section("simulation")
        self.random_cfg = deploy_config.section("sim2sim_randomization")
        self.ranges = self.random_cfg["train_match_ranges"]

    @staticmethod
    def _rng(seed: int, episode_index: int, category: str) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(episode_index), _CATEGORY_IDS[category]]
            )
        )

    @staticmethod
    def _scaled(nominal: np.ndarray, sampled: np.ndarray, scale: float) -> np.ndarray:
        return nominal + scale * (sampled - nominal)

    @staticmethod
    def _signed_uniform(
        rng: np.random.Generator, minimum_abs: float, maximum_abs: float, size: int
    ) -> np.ndarray:
        magnitude = rng.uniform(minimum_abs, maximum_abs, size=size)
        sign = rng.choice(np.asarray((-1.0, 1.0)), size=size)
        return magnitude * sign

    def sample(
        self, profile: str, seed: int, episode_index: int = 0
    ) -> ScenarioSample:
        if profile not in PROFILE_NAMES:
            raise ValueError(
                f"randomization profile must be one of {PROFILE_NAMES}, got {profile!r}"
            )
        profile_cfg = self.random_cfg["profiles"][profile]
        scale = float(profile_cfg["scale"])
        reset_rng = self._rng(seed, episode_index, "reset")
        physics_rng = self._rng(seed, episode_index, "physics")
        actuator_rng = self._rng(seed, episode_index, "actuator")
        sensor_seed = int(
            self._rng(seed, episode_index, "sensor").integers(0, 2**31)
        )
        disturbance_seed = int(
            self._rng(seed, episode_index, "disturbance").integers(0, 2**31)
        )

        nominal_robot_position = np.asarray(
            self.sim_cfg["robot_initial_position"], dtype=np.float64
        )
        nominal_robot_quat = np.asarray(
            self.sim_cfg["robot_initial_quaternion_wxyz"], dtype=np.float64
        )
        nominal_box_size = np.asarray(self.sim_cfg["box_size"], dtype=np.float64)
        nominal_box_position = np.asarray(
            self.sim_cfg["box_initial_position"], dtype=np.float64
        )
        nominal_box_quat = np.asarray(
            self.sim_cfg["box_initial_quaternion_wxyz"], dtype=np.float64
        )
        nominal_goal = np.asarray(self.sim_cfg["goal_position"], dtype=np.float64)
        nominal_density = float(self.sim_cfg["box_density"])

        joint_offset = reset_rng.uniform(
            *self.ranges["joint_position_offset"], size=ACTION_DIM
        )
        joint_position = np.asarray(DEFAULT_DOF_POS) + scale * joint_offset
        root_linear_velocity = scale * reset_rng.uniform(
            *self.ranges["root_linear_velocity"], size=3
        )
        root_angular_velocity = scale * reset_rng.uniform(
            *self.ranges["root_angular_velocity"], size=3
        )
        train_box_scale = np.asarray(
            [
                reset_rng.uniform(*self.ranges["box_scale_x"]),
                reset_rng.uniform(*self.ranges["box_scale_y"]),
                reset_rng.uniform(*self.ranges["box_scale_z"]),
            ]
        )
        box_scale = 1.0 + scale * (train_box_scale - 1.0)
        box_size = nominal_box_size * box_scale
        train_density = reset_rng.uniform(*self.ranges["box_density"])
        box_density = nominal_density + scale * (train_density - nominal_density)

        train_box_xy = self._signed_uniform(
            reset_rng,
            float(self.ranges["box_xy_abs"][0]),
            float(self.ranges["box_xy_abs"][1]),
            2,
        )
        train_box_z = max(
            float(reset_rng.uniform(*self.ranges["box_z"])),
            0.5 * float(box_size[2]),
        ) + 0.01
        train_box_position = np.asarray(
            (train_box_xy[0], train_box_xy[1], train_box_z)
        )
        box_position = self._scaled(
            nominal_box_position, train_box_position, scale
        )
        train_box_yaw = float(reset_rng.uniform(0.0, 2.0 * math.pi))
        box_quaternion = _normalize_quat_wxyz(
            _quat_mul_wxyz(nominal_box_quat, _yaw_quat(scale * train_box_yaw))
        )

        direction_to_robot = nominal_robot_position[:2] - box_position[:2]
        base_angle = math.atan2(direction_to_robot[1], direction_to_robot[0])
        bearing = math.radians(
            float(reset_rng.uniform(*self.ranges["goal_bearing_abs_deg"]))
        )
        bearing *= float(reset_rng.choice((-1.0, 1.0)))
        goal_distance = float(reset_rng.uniform(*self.ranges["goal_distance"]))
        train_goal_xy = box_position[:2] + goal_distance * np.asarray(
            (math.cos(base_angle + bearing), math.sin(base_angle + bearing))
        )
        train_goal_z = max(
            float(reset_rng.uniform(*self.ranges["goal_z"])),
            0.5 * float(box_size[2])
            + float(self.sim_cfg["source_platform"]["size"][2]),
        )
        train_goal = np.asarray((train_goal_xy[0], train_goal_xy[1], train_goal_z))
        goal_position = self._scaled(nominal_goal, train_goal, scale)

        nominal_friction = float(
            self.random_cfg.get(
                "nominal_robot_friction",
                self.model_default_friction_fallback(),
            )
        )
        sampled_friction = float(
            physics_rng.uniform(*self.ranges["robot_friction"])
        )
        robot_friction = nominal_friction + scale * (
            sampled_friction - nominal_friction
        )
        mass_low, mass_high = self.ranges["link_mass_scale"]
        link_mass_range = [
            1.0 + scale * (float(mass_low) - 1.0),
            1.0 + scale * (float(mass_high) - 1.0),
        ]
        payload = scale * float(
            physics_rng.uniform(*self.ranges["torso_payload_mass"])
        )
        com = scale * physics_rng.uniform(
            *self.ranges["torso_com_displacement"], size=3
        )
        kp = 1.0 + scale * (
            actuator_rng.uniform(*self.ranges["kp_factor"], size=ACTION_DIM) - 1.0
        )
        kd = 1.0 + scale * (
            actuator_rng.uniform(*self.ranges["kd_factor"], size=ACTION_DIM) - 1.0
        )
        motor = 1.0 + scale * (
            actuator_rng.uniform(
                *self.ranges["motor_strength"], size=ACTION_DIM
            )
            - 1.0
        )
        torque_bias = scale * actuator_rng.uniform(
            *self.ranges["torque_bias_fraction"], size=ACTION_DIM
        )
        max_delay = int(profile_cfg["max_action_delay_steps"])
        delay = int(actuator_rng.integers(0, max_delay + 1)) if max_delay else 0
        noise = self.ranges["observation_noise"]
        disturbance_force = (
            scale * float(self.ranges["disturbance_force"])
            if bool(profile_cfg.get("disturbance", False))
            else 0.0
        )
        approximation = (
            "none"
            if profile == "nominal"
            else (
                "train_match_approximation"
                if profile == "train_match"
                else "light_train_envelope_approximation"
            )
        )
        link_seed = int(physics_rng.integers(0, 2**31))
        return ScenarioSample(
            format_version=1,
            profile=profile,
            seed=int(seed),
            episode_index=int(episode_index),
            approximation=approximation,
            robot_position=nominal_robot_position.tolist(),
            robot_quaternion_wxyz=nominal_robot_quat.tolist(),
            root_linear_velocity=root_linear_velocity.tolist(),
            root_angular_velocity=root_angular_velocity.tolist(),
            joint_position=joint_position.tolist(),
            box_size=box_size.tolist(),
            box_density=float(box_density),
            box_position=box_position.tolist(),
            box_quaternion_wxyz=box_quaternion.tolist(),
            goal_position=goal_position.tolist(),
            robot_friction=float(robot_friction),
            link_mass_scale_seed=link_seed,
            link_mass_scale_range=link_mass_range,
            torso_payload_mass=float(payload),
            torso_com_displacement=com.tolist(),
            kp_factors=kp.tolist(),
            kd_factors=kd.tolist(),
            motor_strength=motor.tolist(),
            torque_bias_fraction=torque_bias.tolist(),
            action_delay_steps=delay,
            sensor_seed=sensor_seed,
            gyro_noise=scale * float(noise["gyro"]),
            gravity_rotation_noise=scale * float(noise["gravity_rotation"]),
            joint_pos_noise=scale * float(noise["joint_position"]),
            joint_vel_noise=scale * float(noise["joint_velocity"]),
            endpoint_noise=scale * float(noise["endpoint"]),
            task_position_noise=scale * float(noise["task_position"]),
            task_orientation_noise_rad=scale
            * math.radians(float(noise["task_orientation_deg"])),
            disturbance_seed=disturbance_seed,
            disturbance_interval_policy_steps=int(
                self.ranges["disturbance_interval_policy_steps"]
            ),
            disturbance_force=disturbance_force,
        )

    def model_default_friction_fallback(self) -> float:
        return float(self.sim_cfg.get("default_robot_friction", 0.8))
