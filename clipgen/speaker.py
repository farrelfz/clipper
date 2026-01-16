from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .face import TrackResult
from .logging import JsonlLogger
from .vad import SpeechWindow


@dataclass
class SpeakerChunk:
    start: float
    end: float
    track_id: Optional[int]


@dataclass
class TrackMotion:
    mouth_motion: float
    presence: float
    jitter: float


def _bbox_to_pixels(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )


def _motion_energy(prev: np.ndarray, current: np.ndarray) -> float:
    diff = cv2.absdiff(prev, current)
    return float(np.mean(diff))


def _chunk_range(total_duration: float, chunk_size: float) -> List[Tuple[float, float]]:
    chunks = []
    t = 0.0
    while t < total_duration:
        end = min(total_duration, t + chunk_size)
        chunks.append((t, end))
        t = end
    return chunks


def infer_speaker_timeline(
    video_path: Path,
    track_result: TrackResult,
    speech_windows: List[SpeechWindow],
    chunk_size: float,
    hysteresis_ms: int,
    confidence_threshold: float,
    logger: JsonlLogger,
) -> List[SpeakerChunk]:
    logger.log("speaker", "start")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or track_result.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps else 0.0

    speech_ranges = [(win.start, win.end) for win in speech_windows]

    chunks = _chunk_range(total_duration, chunk_size)
    track_scores: Dict[int, Dict[int, TrackMotion]] = {}

    prev_frame = None
    frame_idx = 0
    chunk_index = 0
    current_chunk = chunks[chunk_index] if chunks else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_s = frame_idx / fps if fps else 0.0
        if current_chunk and time_s > current_chunk[1] and chunk_index < len(chunks) - 1:
            chunk_index += 1
            current_chunk = chunks[chunk_index]
        if prev_frame is None:
            prev_frame = frame
            frame_idx += 1
            continue

        if current_chunk and current_chunk[0] <= time_s <= current_chunk[1]:
            for track in track_result.tracks:
                bbox = track.detections.get(round(time_s, 2))
                if bbox is None:
                    continue
                x1, y1, x2, y2 = _bbox_to_pixels(bbox, track_result.width, track_result.height)
                if x2 <= x1 or y2 <= y1:
                    continue
                mouth_y1 = y1 + int((y2 - y1) * 0.55)
                mouth_roi_prev = prev_frame[mouth_y1:y2, x1:x2]
                mouth_roi = frame[mouth_y1:y2, x1:x2]
                if mouth_roi.size == 0 or mouth_roi_prev.size == 0:
                    continue
                motion = _motion_energy(mouth_roi_prev, mouth_roi)
                track_scores.setdefault(track.track_id, {})[chunk_index] = TrackMotion(
                    mouth_motion=motion,
                    presence=1.0,
                    jitter=0.0,
                )

        prev_frame = frame
        frame_idx += 1

    cap.release()

    # Compute jitter per track based on bbox center deltas.
    for track in track_result.tracks:
        times = sorted(track.detections.keys())
        centers = []
        for t in times:
            x1, y1, x2, y2 = track.detections[t]
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        if len(centers) < 2:
            continue
        jitter = float(np.mean([abs(centers[i][0] - centers[i - 1][0]) + abs(centers[i][1] - centers[i - 1][1]) for i in range(1, len(centers))]))
        for chunk_idx in track_scores.get(track.track_id, {}):
            track_scores[track.track_id][chunk_idx].jitter = jitter

    timeline: List[SpeakerChunk] = []
    last_track: Optional[int] = None
    last_switch_time = 0.0
    for idx, chunk in enumerate(chunks):
        chunk_start, chunk_end = chunk
        in_speech = any(
            win_start <= chunk_end and win_end >= chunk_start for win_start, win_end in speech_ranges
        )
        if not in_speech:
            timeline.append(SpeakerChunk(start=chunk_start, end=chunk_end, track_id=None))
            continue
        scores: Dict[int, float] = {}
        for track_id, per_chunk in track_scores.items():
            if idx not in per_chunk:
                continue
            motion = per_chunk[idx].mouth_motion
            presence = per_chunk[idx].presence
            jitter = per_chunk[idx].jitter
            scores[track_id] = 1.2 * motion + 0.4 * presence - 0.6 * jitter
        if not scores:
            timeline.append(SpeakerChunk(start=chunk_start, end=chunk_end, track_id=last_track))
            continue
        best_track = max(scores, key=scores.get)
        best_score = scores[best_track]
        margin = best_score - sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else best_score
        if last_track is None:
            last_track = best_track
            last_switch_time = chunk_start
        else:
            elapsed_ms = (chunk_start - last_switch_time) * 1000
            if best_track != last_track and margin >= confidence_threshold and elapsed_ms >= hysteresis_ms:
                last_track = best_track
                last_switch_time = chunk_start
        timeline.append(SpeakerChunk(start=chunk_start, end=chunk_end, track_id=last_track))

    logger.log("speaker", "complete", chunks=len(timeline))
    return timeline
