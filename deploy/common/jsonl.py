"""Small optional JSONL recorder for deployment smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
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
    def __init__(
        self, path: str | Path | None, *, flush_every: int = 100
    ) -> None:
        self.path = None if path is None else Path(path).resolve()
        self._stream = None
        self._writes_since_flush = 0
        self._queue: queue.Queue[Any] | None = None
        self._worker: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._sentinel = object()
        self.flush_every = max(1, int(flush_every))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open(
                "w", encoding="utf-8", buffering=1024 * 1024
            )
            self._queue = queue.Queue(maxsize=4096)
            self._worker = threading.Thread(
                target=self._writer_loop,
                name=f"jsonl:{self.path.name}",
                daemon=True,
            )
            self._worker.start()

    def _writer_loop(self) -> None:
        assert self._queue is not None
        assert self._stream is not None
        while True:
            record = self._queue.get()
            try:
                if record is self._sentinel:
                    return
                self._stream.write(
                    json.dumps(
                        _json_value(record), separators=(",", ":")
                    )
                    + "\n"
                )
                self._writes_since_flush += 1
                if self._writes_since_flush >= self.flush_every:
                    self._stream.flush()
                    self._writes_since_flush = 0
            except BaseException as exc:
                self._worker_error = exc
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("JSONL writer thread failed") from self._worker_error

    def write(self, record: dict[str, Any]) -> None:
        if self._queue is not None:
            self._raise_worker_error()
            self._queue.put(record)

    def flush(self) -> None:
        if self._queue is not None:
            self._queue.join()
            self._raise_worker_error()
        if self._stream is not None:
            self._stream.flush()
            self._writes_since_flush = 0

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def close(self) -> None:
        if self._stream is not None:
            assert self._queue is not None
            assert self._worker is not None
            self._queue.put(self._sentinel)
            self._queue.join()
            self._worker.join(timeout=5.0)
            self._raise_worker_error()
            self._stream.flush()
            self._stream.close()
            self._stream = None
            self._queue = None
            self._worker = None
