"""Name-based robot description and mapping validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .constants import ACTION_DIM, JOINT_NAMES


@dataclass(frozen=True, slots=True)
class RobotDescription:
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effort_limits: np.ndarray

    @classmethod
    def from_urdf(
        cls, path: str | Path, joint_names: tuple[str, ...] = JOINT_NAMES
    ) -> "RobotDescription":
        root = ET.parse(Path(path)).getroot()
        joints = {
            joint.attrib["name"]: joint
            for joint in root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        }
        missing = [name for name in joint_names if name not in joints]
        extra = [name for name in joints if name not in joint_names]
        if missing or extra:
            raise ValueError(
                f"URDF policy joint mismatch: missing={missing}, unexpected={extra}"
            )

        lower: list[float] = []
        upper: list[float] = []
        effort: list[float] = []
        for name in joint_names:
            limit = joints[name].find("limit")
            if limit is None:
                raise ValueError(f"joint {name} has no <limit>")
            lower.append(float(limit.attrib["lower"]))
            upper.append(float(limit.attrib["upper"]))
            effort.append(float(limit.attrib["effort"]))
        return cls(
            joint_names=tuple(joint_names),
            lower_limits=np.asarray(lower, dtype=np.float64),
            upper_limits=np.asarray(upper, dtype=np.float64),
            effort_limits=np.asarray(effort, dtype=np.float64),
        )

    def assert_runtime_names(
        self, runtime_joint_names: list[str] | tuple[str, ...]
    ) -> None:
        runtime = tuple(runtime_joint_names)
        if set(runtime) != set(self.joint_names):
            missing = sorted(set(self.joint_names) - set(runtime))
            extra = sorted(set(runtime) - set(self.joint_names))
            raise ValueError(f"runtime joint mismatch: missing={missing}, extra={extra}")

    def runtime_indices(
        self, runtime_joint_names: list[str] | tuple[str, ...]
    ) -> np.ndarray:
        self.assert_runtime_names(runtime_joint_names)
        lookup = {name: index for index, name in enumerate(runtime_joint_names)}
        return np.asarray([lookup[name] for name in self.joint_names], dtype=np.int64)


def validate_motor_mapping(policy_to_motor: list[int] | tuple[int, ...]) -> np.ndarray:
    mapping = np.asarray(policy_to_motor, dtype=np.int64)
    if mapping.shape != (ACTION_DIM,):
        raise ValueError(f"motor mapping must have shape ({ACTION_DIM},)")
    if sorted(mapping.tolist()) != list(range(ACTION_DIM)):
        raise ValueError("motor mapping must be a permutation of 0..28")
    return mapping
