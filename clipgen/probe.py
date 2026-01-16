from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from clipgen.utils import read_json, run_command, write_json


def probe_video(path: Path, output_path: Path) -> Dict[str, Any]:
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
    result = run_command(cmd)
    payload = result.stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload)
    return read_probe(output_path)


def read_probe(output_path: Path) -> Dict[str, Any]:
    return read_json(output_path)


def get_video_stream(probe: Dict[str, Any]) -> Dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("No video stream found.")


def get_audio_stream(probe: Dict[str, Any]) -> Dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    raise ValueError("No audio stream found.")


def estimate_fps(video_stream: Dict[str, Any]) -> float:
    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if not rate:
        return 30.0
    num, denom = rate.split("/")
    if float(denom) == 0:
        return 30.0
    return float(num) / float(denom)


def determine_target_fps(source_fps: float, fps_target: int, fps_fallback: int) -> int:
    if source_fps >= 50:
        return min(60, round(source_fps))
    if source_fps < 25:
        return fps_fallback
    return 30


def write_probe_summary(probe: Dict[str, Any], output_path: Path) -> None:
    video = get_video_stream(probe)
    audio = get_audio_stream(probe)
    summary = {
        "duration": float(probe.get("format", {}).get("duration", 0)),
        "video": {
            "width": video.get("width"),
            "height": video.get("height"),
            "codec": video.get("codec_name"),
            "fps": estimate_fps(video),
        },
        "audio": {
            "codec": audio.get("codec_name"),
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
        },
    }
    write_json(output_path, summary)

