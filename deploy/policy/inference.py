"""ONNX Runtime wrapper with strict actor interface validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deploy.common.constants import ACTION_DIM, ACTOR_OBS_DIM


class OnnxActor:
    def __init__(self, path: str | Path, *, provider: str = "CPUExecutionProvider"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required by the deployment policy process"
            ) from exc
        self.path = Path(path).resolve()
        self.session = ort.InferenceSession(str(self.path), providers=[provider])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs":
            raise ValueError(f"expected one ONNX input named actor_obs, got {inputs}")
        if len(outputs) != 1 or outputs[0].name != "actions":
            raise ValueError(f"expected one ONNX output named actions, got {outputs}")

    def __call__(self, actor_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(actor_obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != ACTOR_OBS_DIM:
            raise ValueError(
                f"actor_obs must have shape (batch, {ACTOR_OBS_DIM}), got {obs.shape}"
            )
        actions = self.session.run(["actions"], {"actor_obs": obs})[0]
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (obs.shape[0], ACTION_DIM):
            raise ValueError(f"ONNX actor returned unexpected shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("ONNX actor returned NaN or Inf")
        return actions
