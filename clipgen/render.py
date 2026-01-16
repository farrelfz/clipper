from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from .config import RenderConfig
from .logging import JsonlLogger
from .speaker import SpeakerChunk


@dataclass
class CropKeyframe:
    time_s: float
    center_x: float
    center_y: float


def _target_crop_size(source_w: int, source_h: int) -> Tuple[int, int]:
    target_ratio = 9 / 16
    crop_w = min(source_w, int(source_h * target_ratio))
    crop_h = int(crop_w / target_ratio)
    if crop_h > source_h:
        crop_h = source_h
        crop_w = int(crop_h * target_ratio)
    return crop_w, crop_h


def _smooth_keyframes(keyframes: List[CropKeyframe], alpha: float) -> List[CropKeyframe]:
    if not keyframes:
        return []
    smoothed = [keyframes[0]]
    for kf in keyframes[1:]:
        prev = smoothed[-1]
        smoothed.append(
            CropKeyframe(
                time_s=kf.time_s,
                center_x=prev.center_x * (1 - alpha) + kf.center_x * alpha,
                center_y=prev.center_y * (1 - alpha) + kf.center_y * alpha,
            )
        )
    return smoothed


def _limit_speed(keyframes: List[CropKeyframe], max_delta: float) -> List[CropKeyframe]:
    if not keyframes:
        return []
    limited = [keyframes[0]]
    for kf in keyframes[1:]:
        prev = limited[-1]
        dx = max(-max_delta, min(max_delta, kf.center_x - prev.center_x))
        dy = max(-max_delta, min(max_delta, kf.center_y - prev.center_y))
        limited.append(CropKeyframe(time_s=kf.time_s, center_x=prev.center_x + dx, center_y=prev.center_y + dy))
    return limited


def _build_expression(keyframes: List[CropKeyframe], axis: str) -> str:
    if not keyframes:
        return "0"
    expr = ""
    for idx in range(len(keyframes) - 1):
        start = keyframes[idx]
        end = keyframes[idx + 1]
        value_start = start.center_x if axis == "x" else start.center_y
        value_end = end.center_x if axis == "x" else end.center_y
        segment = (
            f"if(between(t,{start.time_s:.3f},{end.time_s:.3f}),"
            f"{value_start:.3f}+({value_end:.3f}-{value_start:.3f})*(t-{start.time_s:.3f})/"
            f"({end.time_s:.3f}-{start.time_s:.3f}),"
        )
        expr += segment
    last_value = keyframes[-1].center_x if axis == "x" else keyframes[-1].center_y
    expr += f"{last_value:.3f}" + ")" * (len(keyframes) - 1)
    return expr


def build_crop_filter(
    keyframes: List[CropKeyframe],
    crop_w: int,
    crop_h: int,
    source_w: int,
    source_h: int,
) -> str:
    x_expr = _build_expression(keyframes, "x")
    y_expr = _build_expression(keyframes, "y")
    x = f"min(max({x_expr}-{crop_w/2:.1f},0),{source_w - crop_w})"
    y = f"min(max({y_expr}-{crop_h/2:.1f},0),{source_h - crop_h})"
    return f"crop={crop_w}:{crop_h}:x='{x}':y='{y}'"


def build_keyframes(
    speaker_chunks: Iterable[SpeakerChunk],
    track_centers: List[Tuple[float, float]],
    default_center: Tuple[float, float],
) -> List[CropKeyframe]:
    keyframes = []
    for chunk in speaker_chunks:
        center = default_center
        if chunk.track_id and chunk.track_id - 1 < len(track_centers):
            center = track_centers[chunk.track_id - 1]
        keyframes.append(CropKeyframe(time_s=chunk.start, center_x=center[0], center_y=center[1]))
    if not keyframes:
        keyframes.append(CropKeyframe(time_s=0.0, center_x=default_center[0], center_y=default_center[1]))
    return keyframes


def render_clip(
    input_path: Path,
    output_path: Path,
    subtitle_ass: Path | None,
    crop_filter: str,
    render_config: RenderConfig,
    target_fps: int,
    start: float,
    end: float,
    logger: JsonlLogger,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = [crop_filter, f"scale={render_config.width}:{render_config.height}:flags={render_config.scaler}"]
    if render_config.sharpen:
        filters.append("unsharp=3:3:0.5")
    if render_config.denoise:
        filters.append("hqdn3d")
    filters.append(f"fps={target_fps}")
    filters.append(f"format={render_config.pix_fmt}")
    if subtitle_ass:
        filters.append(f"ass={subtitle_ass}")
    vf = ",".join(filters)

    audio_filters = []
    if render_config.loudness_normalize:
        audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    af = ",".join(audio_filters) if audio_filters else None

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        render_config.codec,
        "-preset",
        render_config.preset,
        "-crf",
        str(render_config.crf),
        "-c:a",
        render_config.audio_codec,
        "-ar",
        str(render_config.audio_rate),
        "-b:a",
        render_config.audio_bitrate,
    ]
    if af:
        cmd.extend(["-af", af])
    cmd.append(str(output_path))

    logger.log("render", "start", output=str(output_path))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr}") from exc
    logger.log("render", "complete", output=str(output_path))


def render_thumbnail(input_path: Path, output_path: Path, timestamp: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.2f}",
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
