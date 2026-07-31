"""Engine-independent CarryBox grasp geometry and contact diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .math_utils import quat_rotate_inverse_wxyz, quat_rotate_wxyz


LEFT_HAND_BODY = "left_rubber_hand"
RIGHT_HAND_BODY = "right_rubber_hand"
BOX_BODY = "carry_box"
ANOMALOUS_SELF_CONTACT_PAIRS = frozenset(
    {
        frozenset(("left_rubber_hand", "left_hip_roll_link")),
        frozenset(("left_rubber_hand", "left_hip_yaw_link")),
        frozenset(("right_rubber_hand", "right_hip_roll_link")),
        frozenset(("right_rubber_hand", "right_hip_yaw_link")),
        frozenset(("left_elbow_link", "left_wrist_pitch_link")),
        frozenset(("right_elbow_link", "right_wrist_pitch_link")),
    }
)


def point_to_obb_signed_distance(
    point_world: np.ndarray,
    box_position_world: np.ndarray,
    box_quaternion_wxyz: np.ndarray,
    box_size: np.ndarray,
) -> float:
    """Return Euclidean distance outside an OBB and negative face depth inside."""

    point = np.asarray(point_world, dtype=np.float64)
    center = np.asarray(box_position_world, dtype=np.float64)
    size = np.asarray(box_size, dtype=np.float64)
    if point.shape != (3,) or center.shape != (3,):
        raise ValueError("point and box_position must have shape (3,)")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("box_size must contain three positive finite values")
    local = quat_rotate_inverse_wxyz(
        box_quaternion_wxyz, point - center
    )
    face_delta = np.abs(local) - 0.5 * size
    outside = np.maximum(face_delta, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    return float(np.max(face_delta))


def obb_bottom_height(
    box_position_world: np.ndarray,
    box_quaternion_wxyz: np.ndarray,
    box_size: np.ndarray,
) -> float:
    """Return the lowest world-Z coordinate of an oriented box."""

    center = np.asarray(box_position_world, dtype=np.float64)
    size = np.asarray(box_size, dtype=np.float64)
    if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("invalid box center or size")
    axes_world = np.stack(
        [
            quat_rotate_wxyz(
                box_quaternion_wxyz, np.eye(3, dtype=np.float64)[axis]
            )
            for axis in range(3)
        ],
        axis=1,
    )
    vertical_half_extent = float(
        np.sum(np.abs(axes_world[2, :]) * 0.5 * size)
    )
    return float(center[2] - vertical_half_extent)


def contact_force_norm(contact: Mapping[str, object]) -> float:
    """Extract a comparable force magnitude from an Isaac or MuJoCo record."""

    for key in (
        "force_world",
        "force_contact_frame",
        "estimated_force_world",
    ):
        value = contact.get(key)
        if value is not None:
            vector = np.asarray(value, dtype=np.float64).reshape(-1)
            if vector.size >= 3:
                return float(np.linalg.norm(vector[:3]))
    if contact.get("force_norm") is not None:
        return float(contact["force_norm"])
    if contact.get("normal_force") is not None:
        return abs(float(contact["normal_force"]))
    return 0.0


def contact_pair(contact: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        (str(contact.get("body1", "")), str(contact.get("body2", "")))
    )


def pair_has_contact(
    contacts: Iterable[Mapping[str, object]],
    body_a: str,
    body_b: str,
    *,
    force_threshold: float = 1.0,
) -> bool:
    expected = frozenset((body_a, body_b))
    return any(
        contact_pair(contact) == expected
        and contact_force_norm(contact) > float(force_threshold)
        for contact in contacts
    )


def has_anomalous_self_contact(
    contacts: Iterable[Mapping[str, object]],
    *,
    force_threshold: float = 1.0,
) -> bool:
    return any(
        contact_pair(contact) in ANOMALOUS_SELF_CONTACT_PAIRS
        and contact_force_norm(contact) > float(force_threshold)
        for contact in contacts
    )


def has_hip_or_torso_box_contact(
    contacts: Iterable[Mapping[str, object]],
    *,
    force_threshold: float = 1.0,
) -> bool:
    for contact in contacts:
        pair = contact_pair(contact)
        if BOX_BODY not in pair or contact_force_norm(contact) <= force_threshold:
            continue
        other = next((name for name in pair if name != BOX_BODY), "")
        if "hip" in other or "torso" in other or "pelvis" in other:
            return True
    return False


@dataclass(slots=True)
class GraspTracker:
    """Track the causal grasp metrics used by the sim2sim acceptance gate."""

    force_threshold: float = 1.0
    required_bimanual_seconds: float = 0.20
    required_clearance_m: float = 0.05
    first_left_contact_time: float | None = None
    first_right_contact_time: float | None = None
    first_bimanual_contact_time: float | None = None
    current_bimanual_duration: float = 0.0
    max_bimanual_duration: float = 0.0
    max_box_clearance: float = float("-inf")
    min_left_palm_distance: float = float("inf")
    min_right_palm_distance: float = float("inf")
    anomalous_self_contact_steps: int = 0
    hip_or_torso_box_contact_steps: int = 0
    grasp_success: bool = False
    _previous_time: float | None = None

    def update(
        self,
        *,
        sim_time: float,
        contacts: Iterable[Mapping[str, object]],
        box_clearance: float,
        left_palm_signed_distance: float,
        right_palm_signed_distance: float,
    ) -> dict[str, object]:
        contacts = tuple(contacts)
        now = float(sim_time)
        dt = (
            0.0
            if self._previous_time is None
            else max(0.0, now - self._previous_time)
        )
        self._previous_time = now
        left = pair_has_contact(
            contacts,
            LEFT_HAND_BODY,
            BOX_BODY,
            force_threshold=self.force_threshold,
        )
        right = pair_has_contact(
            contacts,
            RIGHT_HAND_BODY,
            BOX_BODY,
            force_threshold=self.force_threshold,
        )
        if left and self.first_left_contact_time is None:
            self.first_left_contact_time = now
        if right and self.first_right_contact_time is None:
            self.first_right_contact_time = now
        if left and right:
            if self.first_bimanual_contact_time is None:
                self.first_bimanual_contact_time = now
            self.current_bimanual_duration += dt
        else:
            self.current_bimanual_duration = 0.0
        self.max_bimanual_duration = max(
            self.max_bimanual_duration, self.current_bimanual_duration
        )
        self.max_box_clearance = max(
            self.max_box_clearance, float(box_clearance)
        )
        self.min_left_palm_distance = min(
            self.min_left_palm_distance, float(left_palm_signed_distance)
        )
        self.min_right_palm_distance = min(
            self.min_right_palm_distance, float(right_palm_signed_distance)
        )
        anomalous = has_anomalous_self_contact(
            contacts, force_threshold=self.force_threshold
        )
        substitute = has_hip_or_torso_box_contact(
            contacts, force_threshold=self.force_threshold
        )
        self.anomalous_self_contact_steps += int(anomalous)
        self.hip_or_torso_box_contact_steps += int(substitute)
        if (
            self.current_bimanual_duration + 1e-12
            >= self.required_bimanual_seconds
            and self.max_box_clearance >= self.required_clearance_m
            and self.hip_or_torso_box_contact_steps == 0
        ):
            self.grasp_success = True
        return {
            "left_hand_box_contact": left,
            "right_hand_box_contact": right,
            "both_hand_box_contact": left and right,
            "bimanual_contact_duration": self.current_bimanual_duration,
            "anomalous_self_contact": anomalous,
            "hip_or_torso_box_contact": substitute,
            "grasp_success": self.grasp_success,
        }

    def summary(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "first_left_hand_contact_time": self.first_left_contact_time,
            "first_right_hand_contact_time": self.first_right_contact_time,
            "first_bimanual_contact_time": self.first_bimanual_contact_time,
            "max_bimanual_contact_duration": self.max_bimanual_duration,
            "max_box_clearance": finite_or_none(self.max_box_clearance),
            "min_left_palm_box_signed_distance": finite_or_none(
                self.min_left_palm_distance
            ),
            "min_right_palm_box_signed_distance": finite_or_none(
                self.min_right_palm_distance
            ),
            "anomalous_self_contact_steps": self.anomalous_self_contact_steps,
            "hip_or_torso_box_contact_steps": (
                self.hip_or_torso_box_contact_steps
            ),
            "grasp_success": self.grasp_success,
        }
