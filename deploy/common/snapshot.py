"""Strict loader for deterministic Isaac Gym CarryBox parity snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ACTION_DIM, ACTOR_OBS_DIM
from .math_utils import xyzw_to_wxyz


@dataclass(frozen=True, slots=True)
class CarryBoxSnapshot:
    path: Path
    root_position_world: np.ndarray
    root_quaternion_wxyz: np.ndarray
    root_linear_velocity_world: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    end_effector_pos_policy_frame: np.ndarray
    box_position_world: np.ndarray
    box_quaternion_wxyz: np.ndarray
    box_linear_velocity_world: np.ndarray
    box_angular_velocity_world: np.ndarray
    box_size: np.ndarray
    box_mass: float | None
    goal_position_world: np.ndarray
    platform_position_world: np.ndarray | None
    actor_obs: np.ndarray
    current_frame: np.ndarray
    previous_action: np.ndarray
    policy_action: np.ndarray | None
    phase: str

    @classmethod
    def load(cls, path: str | Path) -> "CarryBoxSnapshot":
        snapshot_path = Path(path).expanduser().resolve()
        with np.load(snapshot_path, allow_pickle=False) as data:
            values = {name: np.asarray(data[name]) for name in data.files}

        def vector(name: str, size: int, *, default=None) -> np.ndarray:
            if name not in values:
                if default is None:
                    raise ValueError(f"snapshot is missing required field {name}")
                value = np.asarray(default, dtype=np.float64)
            else:
                value = np.asarray(values[name], dtype=np.float64)
            value = value.reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(
                    f"snapshot {name} must be finite shape ({size},), "
                    f"got {value.shape}"
                )
            return value

        platform = (
            None
            if "platform_position_world" not in values
            else vector("platform_position_world", 3)
        )
        policy_action = (
            None
            if "policy_action" not in values
            else vector("policy_action", ACTION_DIM)
        )
        box_mass = (
            None
            if "box_mass" not in values
            else float(np.asarray(values["box_mass"]).reshape(()))
        )
        if box_mass is not None and (
            not np.isfinite(box_mass) or box_mass <= 0.0
        ):
            raise ValueError("snapshot box_mass must be positive and finite")
        phase_value = values.get("snapshot_phase", np.asarray("default"))
        phase = str(np.asarray(phase_value).reshape(()))
        box_size = vector("box_size", 3)
        if np.any(box_size <= 0.0):
            raise ValueError("snapshot box_size must be positive")
        return cls(
            path=snapshot_path,
            root_position_world=vector("root_position_world", 3),
            root_quaternion_wxyz=xyzw_to_wxyz(
                vector("root_quat_xyzw", 4)
            ),
            root_linear_velocity_world=vector(
                "root_linear_velocity_world", 3, default=np.zeros(3)
            ),
            root_angular_velocity_world=vector(
                "root_angular_velocity_world", 3, default=np.zeros(3)
            ),
            joint_pos=vector("joint_pos", ACTION_DIM),
            joint_vel=vector("joint_vel", ACTION_DIM),
            end_effector_pos_policy_frame=vector(
                "end_effector_pos_policy_frame", 15
            ).reshape(5, 3),
            box_position_world=vector("box_position_world", 3),
            box_quaternion_wxyz=xyzw_to_wxyz(
                vector("box_quat_xyzw", 4)
            ),
            box_linear_velocity_world=vector(
                "box_linear_velocity_world", 3, default=np.zeros(3)
            ),
            box_angular_velocity_world=vector(
                "box_angular_velocity_world", 3, default=np.zeros(3)
            ),
            box_size=box_size,
            box_mass=box_mass,
            goal_position_world=vector("goal_position_world", 3),
            platform_position_world=platform,
            actor_obs=vector("actor_obs", ACTOR_OBS_DIM).astype(np.float32),
            current_frame=vector("current_frame", 123).astype(np.float32),
            previous_action=vector("previous_action", ACTION_DIM),
            policy_action=policy_action,
            phase=phase,
        )

    @property
    def box_density(self) -> float | None:
        if self.box_mass is None:
            return None
        return float(self.box_mass / np.prod(self.box_size))
