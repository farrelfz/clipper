from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, message: str, **fields: Any) -> None:
        entry: Dict[str, Any] = {
            "ts_ms": now_ms(),
            "stage": stage,
            "message": message,
        }
        entry.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_duration(logger: JsonlLogger, stage: str, start_ms: int, **fields: Any) -> None:
    duration_ms = now_ms() - start_ms
    logger.log(stage, "completed", duration_ms=duration_ms, **fields)


def redact_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not headers:
        return {}
    return {key: "<redacted>" for key in headers.keys()}
