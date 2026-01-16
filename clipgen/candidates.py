from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from clipgen.config import ClipSelectionConfig
from clipgen.utils import write_json


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    tokens: List[str]


def build_candidates(transcript: Dict[str, object], config: ClipSelectionConfig, output_path: Path) -> List[Candidate]:
    segments = transcript.get("segments", [])
    candidates: List[Candidate] = []
    for segment in segments:
        text = segment["text"].strip()
        start = float(segment["start"])
        end = float(segment["end"])
        tokens = text.split()
        if not tokens:
            continue
        candidates.append(Candidate(start=start, end=end, text=text, tokens=tokens))
    capped = _expand_candidates(candidates, config)
    write_json(
        output_path,
        {
            "candidates": [
                {"start": cand.start, "end": cand.end, "text": cand.text, "tokens": cand.tokens}
                for cand in capped
            ]
        },
    )
    return capped


def _expand_candidates(base: List[Candidate], config: ClipSelectionConfig) -> List[Candidate]:
    expanded: List[Candidate] = []
    for i in range(len(base)):
        for j in range(i, min(i + 6, len(base))):
            start = base[i].start
            end = base[j].end
            text = " ".join(part.text for part in base[i : j + 1])
            tokens = [token for part in base[i : j + 1] for token in part.tokens]
            if config.avoid_mid_sentence_cut and not _ends_clean(text):
                continue
            expanded.append(Candidate(start=start, end=end, text=text, tokens=tokens))
    return _filter_by_duration(expanded, config)


def _ends_clean(text: str) -> bool:
    return text.strip().endswith((".", "!", "?", "\""))


def _filter_by_duration(candidates: List[Candidate], config: ClipSelectionConfig) -> List[Candidate]:
    filtered: List[Candidate] = []
    max_allowed = max(config.tiktok.max_seconds, config.shorts.max_seconds, config.reels.max_seconds)
    min_allowed = min(config.tiktok.min_seconds, config.shorts.min_seconds, config.reels.min_seconds)
    for candidate in candidates:
        duration = candidate.end - candidate.start
        if duration < min_allowed or duration > max_allowed:
            continue
        filtered.append(candidate)
    return filtered
