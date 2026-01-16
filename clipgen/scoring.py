from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .candidates import Candidate


@dataclass
class CandidateScore:
    candidate: Candidate
    score: float
    features: Dict[str, float]


FILLER_WORDS = {"um", "uh", "like", "you", "know", "so"}


PLATFORM_WEIGHTS = {
    "tiktok": {
        "hook_strength": 0.35,
        "keyword_hits": 0.2,
        "audio_energy": 0.15,
        "speaking_rate": 0.15,
        "structure": 0.1,
        "novelty": 0.05,
        "penalty": -0.2,
    },
    "shorts": {
        "hook_strength": 0.2,
        "keyword_hits": 0.2,
        "audio_energy": 0.1,
        "speaking_rate": 0.15,
        "structure": 0.25,
        "novelty": 0.1,
        "penalty": -0.2,
    },
    "reels": {
        "hook_strength": 0.15,
        "keyword_hits": 0.15,
        "audio_energy": 0.1,
        "speaking_rate": 0.1,
        "structure": 0.3,
        "novelty": 0.15,
        "penalty": -0.2,
    },
}


def _count_keywords(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def _hook_strength(text: str) -> float:
    words = text.split()
    first = " ".join(words[:10])
    return float(sum(char.isdigit() for char in first) + sum(word.istitle() for word in words[:5]))


def _speaking_rate(text: str, duration: float) -> float:
    words = len(text.split())
    return words / duration if duration else 0.0


def _structure_score(text: str) -> float:
    return 1.0 if text.strip().endswith((".", "!", "?")) else 0.4


def _penalty(text: str) -> float:
    words = [w.strip(",.?!").lower() for w in text.split()]
    if not words:
        return 0.0
    filler = sum(1 for w in words if w in FILLER_WORDS)
    return filler / len(words)


def score_candidates(
    candidates: List[Candidate],
    keywords: List[str],
    platform: str,
) -> List[CandidateScore]:
    weights = PLATFORM_WEIGHTS[platform]
    scores: List[CandidateScore] = []
    for cand in candidates:
        duration = cand.end - cand.start
        features = {
            "hook_strength": _hook_strength(cand.text),
            "keyword_hits": float(_count_keywords(cand.text, keywords)),
            "audio_energy": 0.5,
            "speaking_rate": _speaking_rate(cand.text, duration),
            "structure": _structure_score(cand.text),
            "novelty": 0.0,
            "penalty": _penalty(cand.text),
        }
        score = sum(features[key] * weight for key, weight in weights.items())
        scores.append(CandidateScore(candidate=cand, score=score, features=features))

    scores.sort(key=lambda item: item.score, reverse=True)
    return scores


def select_with_overlap_guard(
    ranked: List[CandidateScore],
    existing: List[Candidate],
    overlap_max_ratio: float,
    limit: int,
) -> List[CandidateScore]:
    selected: List[CandidateScore] = []
    for item in ranked:
        candidate = item.candidate
        overlaps = False
        for chosen in existing + [sel.candidate for sel in selected]:
            inter = max(0.0, min(candidate.end, chosen.end) - max(candidate.start, chosen.start))
            min_dur = min(candidate.end - candidate.start, chosen.end - chosen.start)
            ratio = inter / min_dur if min_dur else 0.0
            if ratio > overlap_max_ratio:
                overlaps = True
                break
        if overlaps:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def build_export_plan(
    candidates: List[Candidate],
    keywords: Dict[str, List[str]],
    overlap_max_ratio: float,
    clips_per_platform: int,
) -> Dict[str, List[CandidateScore]]:
    selected: Dict[str, List[CandidateScore]] = {}
    global_selected: List[Candidate] = []
    for platform in ["tiktok", "shorts", "reels"]:
        ranked = score_candidates(candidates, keywords.get(platform, []), platform)
        picks = select_with_overlap_guard(
            ranked,
            existing=global_selected,
            overlap_max_ratio=overlap_max_ratio,
            limit=clips_per_platform,
        )
        selected[platform] = picks
        global_selected.extend([pick.candidate for pick in picks])
    return selected
