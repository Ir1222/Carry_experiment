"""Quaternion helpers with an explicit WXYZ convention."""

from __future__ import annotations

import numpy as np


def normalize_quat_wxyz(quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("zero-norm quaternion")
    return quat / norm


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    vector = np.asarray(vector, dtype=np.float64)
    q_vec = quat[..., 1:]
    q_w = quat[..., :1]
    first_cross = np.cross(q_vec, vector)
    return vector + 2.0 * (q_w * first_cross + np.cross(q_vec, first_cross))


def quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_rotate_wxyz(quat_conjugate_wxyz(normalize_quat_wxyz(quat)), vector)


def quat_relative_wxyz(parent_world: np.ndarray, child_world: np.ndarray) -> np.ndarray:
    return normalize_quat_wxyz(
        quat_multiply_wxyz(
            quat_conjugate_wxyz(normalize_quat_wxyz(parent_world)),
            normalize_quat_wxyz(child_world),
        )
    )


def quat_to_tan_norm_wxyz(quat: np.ndarray) -> np.ndarray:
    """Match PhysHSI ``quat_to_tan_norm``: rotated local X and local Z axes."""

    quat = normalize_quat_wxyz(quat)
    tangent = quat_rotate_wxyz(quat, np.array([1.0, 0.0, 0.0]))
    normal = quat_rotate_wxyz(quat, np.array([0.0, 0.0, 1.0]))
    return np.concatenate((tangent, normal), axis=-1)


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat[..., (3, 0, 1, 2)]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat[..., (1, 2, 3, 0)]
