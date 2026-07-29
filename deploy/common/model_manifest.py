"""Strict model identity and actor-contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    FRAME_OBS_DIM,
    HISTORY_LENGTH,
    JOINT_NAMES,
)
from .transport import VERSION as TRANSPORT_VERSION


MANIFEST_FORMAT_VERSION = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"model manifest not found: {manifest_path}; re-export the actor"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"model manifest must be a JSON object: {manifest_path}")
    return value


def validate_model_manifest(
    manifest_path: str | Path,
    onnx_path: str | Path,
    *,
    profile: str,
    policy_frame: str,
    action_scale: float,
    physics_hz: int,
    policy_hz: int,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_model_manifest(manifest_path)
    expected_scalars = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "profile": str(profile),
        "policy_frame": str(policy_frame),
        "transport_protocol_version": TRANSPORT_VERSION,
        "action_scale": float(action_scale),
        "physics_hz": int(physics_hz),
        "policy_hz": int(policy_hz),
    }
    for key, expected in expected_scalars.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(
                f"manifest {key}={actual!r}, expected {expected!r}"
            )

    dimensions = manifest.get("dimensions")
    expected_dimensions = {
        "frame_obs": FRAME_OBS_DIM,
        "history_length": HISTORY_LENGTH,
        "actor_obs": ACTOR_OBS_DIM,
        "actions": ACTION_DIM,
    }
    if dimensions != expected_dimensions:
        raise ValueError(
            f"manifest dimensions={dimensions!r}, "
            f"expected {expected_dimensions!r}"
        )
    if tuple(manifest.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("manifest joint_names do not match the policy contract")

    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX actor not found: {onnx_path}")
    actual_onnx_sha = sha256_file(onnx_path)
    if manifest.get("onnx_sha256") != actual_onnx_sha:
        raise ValueError(
            "ONNX SHA256 does not match its manifest: "
            f"{actual_onnx_sha} != {manifest.get('onnx_sha256')}"
        )

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).resolve()
        if checkpoint.is_file():
            actual_checkpoint_sha = sha256_file(checkpoint)
            if manifest.get("checkpoint_sha256") != actual_checkpoint_sha:
                raise ValueError(
                    "checkpoint SHA256 does not match its manifest: "
                    f"{actual_checkpoint_sha} != "
                    f"{manifest.get('checkpoint_sha256')}"
                )
    return manifest


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": manifest["profile"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "onnx_sha256": manifest["onnx_sha256"],
        "policy_frame": manifest["policy_frame"],
        "transport_protocol_version": manifest[
            "transport_protocol_version"
        ],
    }
