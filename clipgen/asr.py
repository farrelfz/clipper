from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .logging import JsonlLogger


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: List[Word]


def transcribe_audio(path: Path, model_size: str, logger: JsonlLogger) -> List[Segment]:
    logger.log("asr", "start", model=model_size)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper not installed.") from exc

    model = WhisperModel(model_size, compute_type="int8")
    segments, _info = model.transcribe(str(path), word_timestamps=True)
    results: List[Segment] = []
    for seg in segments:
        words = [Word(text=word.word.strip(), start=word.start, end=word.end) for word in seg.words]
        results.append(Segment(text=seg.text.strip(), start=seg.start, end=seg.end, words=words))
    logger.log("asr", "complete", segments=len(results))
    return results
