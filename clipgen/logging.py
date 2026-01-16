from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LogEvent:
    stage: str
    message: str
    metrics: Dict[str, Any]
    timestamp: float
    duration_s: Optional[float] = None


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, message: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        event = LogEvent(
            stage=stage,
            message=message,
            metrics=metrics or {},
            timestamp=time.time(),
        )
        self._write(event)

    def timed(self, stage: str, message: str, metrics: Optional[Dict[str, Any]] = None):
        return _Timer(self, stage, message, metrics or {})

    def _write(self, event: LogEvent) -> None:
        payload = {
            "stage": event.stage,
            "message": event.message,
            "metrics": event.metrics,
            "timestamp": event.timestamp,
        }
        if event.duration_s is not None:
            payload["duration_s"] = event.duration_s
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class _Timer:
    def __init__(self, logger: JsonlLogger, stage: str, message: str, metrics: Dict[str, Any]) -> None:
        self.logger = logger
        self.stage = stage
        self.message = message
        self.metrics = metrics
        self.start = 0.0

    def __enter__(self) -> "_Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.time() - self.start
        event = LogEvent(
            stage=self.stage,
            message=self.message,
            metrics=self.metrics,
            timestamp=time.time(),
            duration_s=duration,
        )
        self.logger._write(event)
