from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from clipgen.utils import write_json


@dataclass
class VADSegment:
    start: float
    end: float


def run_vad(audio_path: Path, output_path: Path, sample_rate: int = 16000) -> List[VADSegment]:
    model = load_silero_vad()
    wav, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    audio = torch.from_numpy(wav)
    speech = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    segments = [
        VADSegment(start=entry["start"] / sample_rate, end=entry["end"] / sample_rate)
        for entry in speech
    ]
    write_json(output_path, {"segments": [segment.__dict__ for segment in segments]})
    return segments
