"""Small optional JSONL recorder for deployment smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonlRecorder:
    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path).resolve()
        self._stream = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        if self._stream is not None:
            self._stream.write(
                json.dumps(_json_value(record), separators=(",", ":")) + "\n"
            )

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
