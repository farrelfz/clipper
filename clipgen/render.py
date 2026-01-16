from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from clipgen.config import RenderConfig
from clipgen.utils import clamp, run_command


@dataclass
class CropKeyframe:
    time: float
    center_x: float
    center_y: float
    scale: float


def build_crop_path(
    detections: List[Dict[str, object]],
    timeline: List[Dict[str, object]],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    safe_margin: float,
    max_pan_speed: float,
    max_zoom_rate: float,
) -> List[CropKeyframe]:
    frames: List[CropKeyframe] = []
    if not detections or not timeline:
        center_x = width / 2
        center_y = height / 2
        frames.append(CropKeyframe(time=clip_start, center_x=center_x, center_y=center_y, scale=1.0))
        frames.append(CropKeyframe(time=clip_end, center_x=center_x, center_y=center_y, scale=1.0))
        return frames
    det_by_track: Dict[int, List[Tuple[float, Tuple[int, int, int, int]]]] = {}
    for det in detections:
        track_id = int(det["track_id"])
        det_by_track.setdefault(track_id, []).append((float(det["time"]), tuple(det["bbox"])))
    for decision in timeline:
        start = float(decision["start"])
        end = float(decision["end"])
        if end < clip_start or start > clip_end:
            continue
        track_id = int(decision["track_id"])
        detections_for_track = det_by_track.get(track_id, [])
        if not detections_for_track:
            continue
        nearest = min(detections_for_track, key=lambda item: abs(item[0] - start))
        bbox = nearest[1]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        face_width = bbox[2] - bbox[0]
        scale = clamp((face_width / width) * (1 + safe_margin), 0.4, 1.0)
        frames.append(CropKeyframe(time=start, center_x=center_x, center_y=center_y, scale=scale))
    if not frames:
        center_x = width / 2
        center_y = height / 2
        frames.append(CropKeyframe(time=clip_start, center_x=center_x, center_y=center_y, scale=1.0))
    frames.sort(key=lambda frame: frame.time)
    if frames[0].time > clip_start:
        frames.insert(0, CropKeyframe(time=clip_start, center_x=frames[0].center_x, center_y=frames[0].center_y, scale=frames[0].scale))
    if frames[-1].time < clip_end:
        frames.append(CropKeyframe(time=clip_end, center_x=frames[-1].center_x, center_y=frames[-1].center_y, scale=frames[-1].scale))
    return _smooth_path(frames, width, height, max_pan_speed, max_zoom_rate)


def _smooth_path(
    frames: List[CropKeyframe],
    width: int,
    height: int,
    max_pan_speed: float,
    max_zoom_rate: float,
) -> List[CropKeyframe]:
    if len(frames) < 2:
        return frames
    smoothed: List[CropKeyframe] = [frames[0]]
    alpha = 0.35
    for prev, current in zip(frames, frames[1:]):
        dt = max(current.time - prev.time, 0.001)
        max_dx = max_pan_speed * width * dt
        max_dy = max_pan_speed * height * dt
        max_dscale = max_zoom_rate * dt
        target_x = prev.center_x + alpha * (current.center_x - prev.center_x)
        target_y = prev.center_y + alpha * (current.center_y - prev.center_y)
        target_scale = prev.scale + alpha * (current.scale - prev.scale)
        dx = clamp(target_x - prev.center_x, -max_dx, max_dx)
        dy = clamp(target_y - prev.center_y, -max_dy, max_dy)
        ds = clamp(target_scale - prev.scale, -max_dscale, max_dscale)
        smoothed.append(
            CropKeyframe(
                time=current.time,
                center_x=prev.center_x + dx,
                center_y=prev.center_y + dy,
                scale=clamp(prev.scale + ds, 0.4, 1.0),
            )
        )
    return smoothed


def render_clip(
    source: Path,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    crop_path: List[CropKeyframe],
    config: RenderConfig,
    subtitle_ass: Path | None,
) -> None:
    duration = clip_end - clip_start
    adjusted = [
        CropKeyframe(
            time=frame.time - clip_start,
            center_x=frame.center_x,
            center_y=frame.center_y,
            scale=frame.scale,
        )
        for frame in crop_path
    ]
    crop_expr_x, crop_expr_y, crop_expr_w, crop_expr_h = _build_crop_expr(
        adjusted, config.width, config.height
    )
    filters = [
        f"crop=w={crop_expr_w}:h={crop_expr_h}:x={crop_expr_x}:y={crop_expr_y}",
        f"scale={config.width}:{config.height}:flags={config.scaler}",
        f"fps={config.fps_target}",
        f"format={config.pix_fmt}",
    ]
    if subtitle_ass and subtitle_ass.exists():
        filters.append(f"ass={subtitle_ass}")
    vf = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-vf",
        vf,
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pix_fmt,
        "-c:a",
        config.audio.codec,
        "-b:a",
        config.audio.bitrate,
        "-ar",
        str(config.audio.sample_rate),
        "-af",
        "loudnorm" if config.audio.loudness_normalize else "anull",
        str(output_path),
    ]
    run_command(cmd)


def extract_thumbnail(source: Path, timestamp: float, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        str(output_path),
    ]
    run_command(cmd)


def _build_crop_expr(path: List[CropKeyframe], width: int, height: int) -> Tuple[str, str, str, str]:
    if len(path) == 1:
        keyframe = path[0]
        crop_w = width * keyframe.scale
        crop_h = height * keyframe.scale
        x = clamp(keyframe.center_x - crop_w / 2, 0, width - crop_w)
        y = clamp(keyframe.center_y - crop_h / 2, 0, height - crop_h)
        return str(x), str(y), str(crop_w), str(crop_h)
    expr_x = _piecewise_expr(path, width, height, axis="x")
    expr_y = _piecewise_expr(path, width, height, axis="y")
    expr_w = _piecewise_expr(path, width, height, axis="w")
    expr_h = _piecewise_expr(path, width, height, axis="h")
    return expr_x, expr_y, expr_w, expr_h


def _piecewise_expr(
    path: List[CropKeyframe],
    width: int,
    height: int,
    axis: str,
) -> str:
    expr = "0"
    for idx in range(len(path) - 1):
        start = path[idx]
        end = path[idx + 1]
        duration = max(end.time - start.time, 0.001)
        if axis == "w":
            start_val = width * start.scale
            end_val = width * end.scale
        elif axis == "h":
            start_val = height * start.scale
            end_val = height * end.scale
        elif axis == "x":
            start_w = width * start.scale
            end_w = width * end.scale
            start_val = clamp(start.center_x - start_w / 2, 0, width - start_w)
            end_val = clamp(end.center_x - end_w / 2, 0, width - end_w)
        else:
            start_h = height * start.scale
            end_h = height * end.scale
            start_val = clamp(start.center_y - start_h / 2, 0, height - start_h)
            end_val = clamp(end.center_y - end_h / 2, 0, height - end_h)
        segment_expr = (
            f"if(between(t,{start.time:.3f},{end.time:.3f}),"
            f"{start_val:.3f}+({end_val:.3f}-{start_val:.3f})*(t-{start.time:.3f})/{duration:.3f},"
        )
        expr = segment_expr + expr + ")"
    return expr
