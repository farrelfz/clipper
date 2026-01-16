from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


class ProbeError(RuntimeError):
    pass


def run_ffprobe(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe not found. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise ProbeError(f"ffprobe failed: {exc.stderr}") from exc
    return json.loads(result.stdout)


def pick_video_stream(probe: Dict[str, Any]) -> Dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ProbeError("No video stream found.")


def estimate_fps(video_stream: Dict[str, Any]) -> float:
    rate = video_stream.get("r_frame_rate", "0/1")
    num, den = rate.split("/")
    try:
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return 0.0


def select_target_fps(source_fps: float) -> int:
    if source_fps >= 50:
        return min(60, round(source_fps))
    if source_fps < 25:
        return 30
    return 30
