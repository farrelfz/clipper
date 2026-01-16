from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .candidates import Candidate
from .face import TrackResult
from .speaker import SpeakerChunk


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def export_tracks(track_result: TrackResult, path: Path) -> None:
    payload = {
        "fps": track_result.fps,
        "width": track_result.width,
        "height": track_result.height,
        "tracks": [
            {
                "track_id": track.track_id,
                "detections": track.detections,
                "scores": track.scores,
            }
            for track in track_result.tracks
        ],
    }
    write_json(path, payload)


def export_speaker_timeline(timeline: Iterable[SpeakerChunk], path: Path) -> None:
    payload = [asdict(chunk) for chunk in timeline]
    write_json(path, payload)


def export_candidates(candidates: Iterable[Candidate], path: Path) -> None:
    payload = [asdict(candidate) for candidate in candidates]
    write_json(path, payload)


def export_plan(plan: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    write_json(path, plan)
