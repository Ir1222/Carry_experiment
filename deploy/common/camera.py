"""D455 camera intrinsics and MuJoCo box projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    vertical_fov_deg: float
    near_m: float
    far_m: float

    @classmethod
    def from_config(
        cls, camera_config: Mapping[str, Any]
    ) -> "CameraIntrinsics":
        return cls(
            width=int(camera_config["width"]),
            height=int(camera_config["height"]),
            vertical_fov_deg=float(
                camera_config["vertical_fov_deg"]
            ),
            near_m=float(camera_config["near_m"]),
            far_m=float(camera_config["far_m"]),
        )

    @property
    def focal_length_px(self) -> float:
        return 0.5 * self.height / math.tan(
            math.radians(self.vertical_fov_deg) / 2.0
        )

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(
            2.0
            * math.atan(
                (self.width / self.height)
                * math.tan(
                    math.radians(self.vertical_fov_deg) / 2.0
                )
            )
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "vertical_fov_deg": self.vertical_fov_deg,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "near_m": self.near_m,
            "far_m": self.far_m,
            "focal_length_px": self.focal_length_px,
        }


def _box_corners_world(
    center_world: np.ndarray,
    rotation_box_to_world: np.ndarray,
    half_size: np.ndarray,
) -> np.ndarray:
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    corners_local = signs * np.asarray(half_size, dtype=np.float64)
    return (
        np.asarray(center_world, dtype=np.float64)
        + corners_local @ np.asarray(
            rotation_box_to_world, dtype=np.float64
        ).T
    )


def project_box_to_camera(
    *,
    camera_position_world: np.ndarray,
    camera_rotation_to_world: np.ndarray,
    box_position_world: np.ndarray,
    box_rotation_to_world: np.ndarray,
    box_half_size: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> dict[str, object]:
    """Project an oriented box into MuJoCo's -Z-forward camera frame."""

    camera_position_world = np.asarray(
        camera_position_world, dtype=np.float64
    )
    camera_rotation_to_world = np.asarray(
        camera_rotation_to_world, dtype=np.float64
    ).reshape(3, 3)
    corners_world = _box_corners_world(
        box_position_world,
        np.asarray(box_rotation_to_world, dtype=np.float64).reshape(3, 3),
        box_half_size,
    )
    corners_camera = (
        corners_world - camera_position_world
    ) @ camera_rotation_to_world
    center_camera = (
        np.asarray(box_position_world, dtype=np.float64)
        - camera_position_world
    ) @ camera_rotation_to_world
    corner_depths = -corners_camera[:, 2]
    center_depth = float(-center_camera[2])
    in_depth = (
        (corner_depths > intrinsics.near_m)
        & (corner_depths < intrinsics.far_m)
    )
    behind_camera = bool(
        np.max(corner_depths) <= intrinsics.near_m
    )

    focal = intrinsics.focal_length_px
    principal = np.asarray(
        (0.5 * intrinsics.width, 0.5 * intrinsics.height),
        dtype=np.float64,
    )

    center_uv: list[float] | None = None
    if center_depth > intrinsics.near_m:
        center_uv_array = principal + np.asarray(
            (
                focal * center_camera[0] / center_depth,
                -focal * center_camera[1] / center_depth,
            )
        )
        center_uv = center_uv_array.tolist()

    bbox_uv: list[float] | None = None
    fully_visible = False
    partially_visible = False
    if np.any(in_depth):
        valid = corners_camera[in_depth]
        depths = corner_depths[in_depth]
        projected = np.column_stack(
            (
                principal[0] + focal * valid[:, 0] / depths,
                principal[1] - focal * valid[:, 1] / depths,
            )
        )
        minimum = np.min(projected, axis=0)
        maximum = np.max(projected, axis=0)
        bbox_uv = [
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ]
        intersects_image = (
            maximum[0] >= 0.0
            and minimum[0] <= intrinsics.width - 1
            and maximum[1] >= 0.0
            and minimum[1] <= intrinsics.height - 1
        )
        fully_visible = bool(
            np.all(in_depth)
            and minimum[0] >= 0.0
            and maximum[0] <= intrinsics.width - 1
            and minimum[1] >= 0.0
            and maximum[1] <= intrinsics.height - 1
        )
        partially_visible = bool(intersects_image and not fully_visible)

    return {
        "box_center_uv": center_uv,
        "box_bbox_uv": bbox_uv,
        "center_depth_m": center_depth,
        "partially_visible": partially_visible,
        "fully_visible": fully_visible,
        "behind_camera": behind_camera,
    }
