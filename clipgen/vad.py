from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .logging import JsonlLogger


@dataclass
class SpeechWindow:
    start: float
    end: float


def run_vad(path: Path, logger: JsonlLogger) -> List[SpeechWindow]:
    logger.log("vad", "start")
    try:
        import torch
        from silero_vad import VADIterator, load_silero_vad
    except ImportError as exc:
        raise RuntimeError("silero-vad not installed.") from exc

    model, utils = load_silero_vad(onnx=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    wav = read_audio(str(path), sampling_rate=16000)
    speech = get_speech_timestamps(wav, model, sampling_rate=16000)
    windows = [SpeechWindow(start=chunk["start"] / 16000, end=chunk["end"] / 16000) for chunk in speech]
    logger.log("vad", "complete", windows=len(windows))
    return windows
