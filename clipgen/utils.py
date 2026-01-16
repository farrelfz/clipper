from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def ensure_ffmpeg() -> None:
    run_command(["ffmpeg", "-version"])
    run_command(["ffprobe", "-version"])


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
