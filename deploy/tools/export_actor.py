"""Export the deterministic CarryBox actor from an rsl_rl checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deploy.common.config import load_deploy_config
from deploy.common.constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    END_EFFECTOR_NAMES,
    FRAME_OBS_DIM,
    HISTORY_LENGTH,
    JOINT_NAMES,
    OBSERVATION_SLICES,
)

ACTOR_KEYS = (
    "actor.0.weight",
    "actor.0.bias",
    "actor.2.weight",
    "actor.2.bias",
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
)


class CarryBoxActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(ACTOR_OBS_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, ACTION_DIM),
        )

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_obs)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "actor.0.weight": (512, ACTOR_OBS_DIM),
        "actor.0.bias": (512,),
        "actor.2.weight": (256, 512),
        "actor.2.bias": (256,),
        "actor.4.weight": (256, 256),
        "actor.4.bias": (256,),
        "actor.6.weight": (ACTION_DIM, 256),
        "actor.6.bias": (ACTION_DIM,),
    }


def load_actor_from_checkpoint(checkpoint_path: str | Path) -> CarryBoxActor:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    state = checkpoint["model_state_dict"]
    expected = _expected_shapes()
    for key, shape in expected.items():
        if key not in state:
            raise ValueError(f"checkpoint is missing {key}")
        if tuple(state[key].shape) != shape:
            raise ValueError(
                f"{key} has shape {tuple(state[key].shape)}, expected {shape}"
            )

    actor = CarryBoxActor().eval()
    actor_state = {key: state[key] for key in ACTOR_KEYS}
    incompatible = actor.load_state_dict(actor_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"actor load failed: {incompatible}")
    return actor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_actor(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path,
    *,
    action_scale: float = 0.25,
    physics_hz: int = 200,
    policy_hz: int = 50,
    verify: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    onnx_path = Path(onnx_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    actor = load_actor_from_checkpoint(checkpoint_path)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, ACTOR_OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["actor_obs"],
        output_names=["actions"],
        dynamic_axes={"actor_obs": {0: "batch"}, "actions": {0: "batch"}},
        dynamo=False,
    )

    max_abs_error: float | None = None
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for export verification; "
                "install deploy/requirements.txt or pass --no-verify"
            ) from exc
        generator = np.random.default_rng(73500)
        sample = generator.standard_normal((8, ACTOR_OBS_DIM)).astype(np.float32)
        with torch.inference_mode():
            torch_output = actor(torch.from_numpy(sample)).numpy()
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run(["actions"], {"actor_obs": sample})[0]
        max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
        if max_abs_error >= 1e-5:
            raise RuntimeError(
                f"ONNX parity failed: max_abs_error={max_abs_error:.8g}"
            )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "input": {"name": "actor_obs", "dtype": "float32", "shape": ["batch", 738]},
        "output": {"name": "actions", "dtype": "float32", "shape": ["batch", 29]},
        "network": {
            "hidden_dims": [512, 256, 256],
            "activation": "elu",
            "deterministic": True,
        },
        "dimensions": {
            "frame_obs": FRAME_OBS_DIM,
            "history_length": HISTORY_LENGTH,
            "actor_obs": ACTOR_OBS_DIM,
            "actions": ACTION_DIM,
        },
        "observation_slices": {
            key: list(value) for key, value in OBSERVATION_SLICES.items()
        },
        "joint_names": list(JOINT_NAMES),
        "end_effector_names": list(END_EFFECTOR_NAMES),
        "action_scale": float(action_scale),
        "physics_hz": int(physics_hz),
        "policy_hz": int(policy_hz),
        "onnx_max_abs_error": max_abs_error,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="deploy/config/g1_carrybox.yaml", help="deployment YAML"
    )
    parser.add_argument("--checkpoint", help="override checkpoint path")
    parser.add_argument("--output", help="override ONNX output path")
    parser.add_argument("--manifest", help="override manifest output path")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    cfg = load_deploy_config(args.config)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else cfg.checkpoint_path
    output = Path(args.output).resolve() if args.output else cfg.onnx_path
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest
        else cfg.resolve_path(cfg.section("policy")["manifest_path"])
    )
    result = export_actor(
        checkpoint,
        output,
        manifest,
        action_scale=float(cfg.section("control")["action_scale"]),
        physics_hz=int(cfg.section("control")["physics_hz"]),
        policy_hz=int(cfg.section("control")["policy_hz"]),
        verify=not args.no_verify,
    )
    print(
        f"Exported {result['onnx']} "
        f"(sha256={result['onnx_sha256']}, parity={result['onnx_max_abs_error']})"
    )


if __name__ == "__main__":
    main()
