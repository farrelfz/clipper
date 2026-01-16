from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from clipgen.config import ClipSelectionConfig, NicheConfig, PLATFORM_WEIGHTS


@dataclass
class ScoredCandidate:
    start: float
    end: float
    text: str
    score: float
    features: Dict[str, float]


def score_candidates(
    candidates: Iterable[Dict[str, object]],
    config: ClipSelectionConfig,
    niche: NicheConfig,
    platform: str,
) -> List[ScoredCandidate]:
    weights = PLATFORM_WEIGHTS[platform]
    scored: List[ScoredCandidate] = []
    for candidate in candidates:
        text = str(candidate["text"])
        tokens = [token.lower() for token in candidate["tokens"]]
        start = float(candidate["start"])
        end = float(candidate["end"])
        duration = end - start
        hook_strength = _hook_strength(tokens)
        keyword_hits = _keyword_hits(tokens, niche)
        audio_energy_peaks = 0.5
        speaking_rate = len(tokens) / max(duration, 0.01)
        structure = _structure_score(text)
        novelty = 0.5
        penalty = _penalty(tokens, text)
        features = {
            "hook_strength": hook_strength,
            "keyword_hits": keyword_hits,
            "audio_energy_peaks": audio_energy_peaks,
            "speaking_rate": speaking_rate,
            "structure": structure,
            "novelty": novelty,
            "penalty": penalty,
        }
        score = (
            weights.hook_strength * hook_strength
            + weights.keyword_hits * keyword_hits
            + weights.audio_energy_peaks * audio_energy_peaks
            + weights.speaking_rate * speaking_rate
            + weights.structure * structure
            + weights.novelty * novelty
            - weights.penalty * penalty
        )
        if not _within_platform(duration, config, platform):
            continue
        scored.append(ScoredCandidate(start=start, end=end, text=text, score=score, features=features))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def select_non_overlapping(
    scored: Dict[str, List[ScoredCandidate]],
    overlap_max_ratio: float,
    clips_per_platform: int,
) -> Dict[str, List[ScoredCandidate]]:
    selected: Dict[str, List[ScoredCandidate]] = {platform: [] for platform in scored}
    chosen: List[ScoredCandidate] = []
    for platform, ranked in scored.items():
        for candidate in ranked:
            if len(selected[platform]) >= clips_per_platform:
                break
            if any(overlap_ratio(candidate, existing) > overlap_max_ratio for existing in chosen):
                continue
            selected[platform].append(candidate)
            chosen.append(candidate)
    return selected


def overlap_ratio(a: ScoredCandidate, b: ScoredCandidate) -> float:
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    if end <= start:
        return 0.0
    intersection = end - start
    shortest = min(a.end - a.start, b.end - b.start)
    if shortest == 0:
        return 0.0
    return intersection / shortest


def _hook_strength(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    target = tokens[: min(8, len(tokens))]
    signal = sum(1 for token in target if token.isdigit() or len(token) > 6)
    return signal / len(target)


def _keyword_hits(tokens: List[str], niche: NicheConfig) -> float:
    if not tokens:
        return 0.0
    keywords = {kw.lower() for kw in niche.keywords + niche.cta_terms}
    hits = sum(1 for token in tokens if token in keywords)
    return hits / len(tokens)


def _structure_score(text: str) -> float:
    if text.endswith((".", "!", "?")):
        return 1.0
    if "," in text:
        return 0.7
    return 0.4


def _penalty(tokens: List[str], text: str) -> float:
    fillers = {"um", "uh", "like", "you", "know"}
    filler_hits = sum(1 for token in tokens if token in fillers)
    return filler_hits / max(1, len(tokens))


def _within_platform(duration: float, config: ClipSelectionConfig, platform: str) -> bool:
    range_cfg = getattr(config, platform)
    return range_cfg.min_seconds <= duration <= range_cfg.max_seconds
