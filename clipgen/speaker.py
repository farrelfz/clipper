from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from clipgen.utils import write_json


@dataclass
class SpeakerDecision:
    start: float
    end: float
    track_id: int
    confidence: float


def build_speaker_timeline(
    detections: List[Dict[str, object]],
    vad_segments: List[Dict[str, float]],
    output_path: Path,
    chunk_size: float,
    switch_hysteresis_s: float,
    confidence_threshold: float,
) -> List[SpeakerDecision]:
    if not detections:
        timeline: List[SpeakerDecision] = []
        write_json(output_path, {"timeline": []})
        return timeline
    by_track: Dict[int, List[Tuple[float, Tuple[int, int, int, int]]]] = {}
    for det in detections:
        track_id = int(det["track_id"])
        by_track.setdefault(track_id, []).append((float(det["time"]), tuple(det["bbox"])))
    max_time = max(time for time, _ in [item for sub in by_track.values() for item in sub])
    speech_windows = [(seg["start"], seg["end"]) for seg in vad_segments]
    def in_speech(t: float) -> bool:
        return any(start <= t <= end for start, end in speech_windows)
    decisions: List[SpeakerDecision] = []
    last_track = None
    last_switch_time = -switch_hysteresis_s
    t = 0.0
    while t <= max_time:
        if not in_speech(t):
            t += chunk_size
            continue
        scores = []
        for track_id, entries in by_track.items():
            chunk_entries = [bbox for time, bbox in entries if t <= time < t + chunk_size]
            if not chunk_entries:
                continue
            presence = len(chunk_entries) / max(1, len(entries))
            jitter = _bbox_jitter(chunk_entries)
            mouth_motion = _bbox_area_variance(chunk_entries)
            score = (0.55 * mouth_motion) + (0.35 * presence) - (0.1 * jitter)
            scores.append((track_id, score))
        if not scores:
            t += chunk_size
            continue
        scores.sort(key=lambda pair: pair[1], reverse=True)
        best_track, best_score = scores[0]
        if last_track is None:
            last_track = best_track
            last_switch_time = t
        else:
            if best_track != last_track and (t - last_switch_time) >= switch_hysteresis_s:
                margin = best_score - scores[1][1] if len(scores) > 1 else best_score
                if margin >= confidence_threshold:
                    last_track = best_track
                    last_switch_time = t
        decisions.append(
            SpeakerDecision(
                start=t,
                end=min(t + chunk_size, max_time),
                track_id=last_track,
                confidence=best_score,
            )
        )
        t += chunk_size
    write_json(
        output_path,
        {
            "timeline": [
                {
                    "start": decision.start,
                    "end": decision.end,
                    "track_id": decision.track_id,
                    "confidence": decision.confidence,
                }
                for decision in decisions
            ]
        },
    )
    return decisions


def _bbox_jitter(boxes: List[Tuple[int, int, int, int]]) -> float:
    if len(boxes) < 2:
        return 0.0
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    jitter = 0.0
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        jitter += abs(x2 - x1) + abs(y2 - y1)
    return jitter / len(centers)


def _bbox_area_variance(boxes: List[Tuple[int, int, int, int]]) -> float:
    if len(boxes) < 2:
        return 0.0
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    mean_area = sum(areas) / len(areas)
    variance = sum((area - mean_area) ** 2 for area in areas) / len(areas)
    return variance / max(mean_area, 1)
