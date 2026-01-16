from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel

from clipgen.utils import write_json


@dataclass
class WordStamp:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[WordStamp]


@dataclass
class Transcript:
    language: str
    segments: List[Segment]


def transcribe(path: Path, output_path: Path, model_name: str = "medium") -> Transcript:
    model = WhisperModel(model_name)
    segments, info = model.transcribe(str(path), word_timestamps=True)
    parsed: List[Segment] = []
    for segment in segments:
        words = [WordStamp(start=w.start, end=w.end, word=w.word) for w in segment.words or []]
        parsed.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
            )
        )
    transcript = Transcript(language=info.language, segments=parsed)
    write_json(output_path, transcript_to_dict(transcript))
    return transcript


def transcript_to_dict(transcript: Transcript) -> Dict[str, object]:
    return {
        "language": transcript.language,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [asdict(word) for word in seg.words],
            }
            for seg in transcript.segments
        ],
    }
