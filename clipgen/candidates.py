from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .asr import Segment
from .vad import SpeechWindow


@dataclass
class Candidate:
    start: float
    end: float
    text: str


def _split_on_pauses(segments: List[Segment], pause_threshold: float = 0.35) -> List[Segment]:
    if not segments:
        return []
    merged: List[Segment] = []
    buffer_words = []
    start = segments[0].start
    for idx, seg in enumerate(segments):
        if buffer_words:
            gap = seg.start - segments[idx - 1].end
        else:
            gap = 0
        if gap >= pause_threshold and buffer_words:
            text = " ".join(word.text for word in buffer_words)
            merged.append(
                Segment(text=text, start=start, end=segments[idx - 1].end, words=buffer_words)
            )
            buffer_words = []
            start = seg.start
        buffer_words.extend(seg.words)
    if buffer_words:
        text = " ".join(word.text for word in buffer_words)
        merged.append(Segment(text=text, start=start, end=segments[-1].end, words=buffer_words))
    return merged


def generate_candidates(
    segments: List[Segment],
    speech_windows: List[SpeechWindow],
    min_s: float,
    max_s: float,
    candidate_pool: int,
) -> List[Candidate]:
    speech_ranges = [(w.start, w.end) for w in speech_windows]
    merged = _split_on_pauses(segments)
    candidates: List[Candidate] = []
    for seg in merged:
        if seg.end - seg.start < min_s:
            continue
        if seg.end - seg.start > max_s:
            # Split long segments into multiple candidates.
            start = seg.start
            while start < seg.end:
                end = min(seg.end, start + max_s)
                text = seg.text
                candidates.append(Candidate(start=start, end=end, text=text))
                start = end
            continue
        candidates.append(Candidate(start=seg.start, end=seg.end, text=seg.text))

    # keep only candidates in speech windows
    filtered: List[Candidate] = []
    for cand in candidates:
        if any(win_start <= cand.end and win_end >= cand.start for win_start, win_end in speech_ranges):
            filtered.append(cand)
    return filtered[:candidate_pool]
