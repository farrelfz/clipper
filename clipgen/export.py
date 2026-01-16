from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from clipgen.utils import write_json


@dataclass
class ExportClip:
    platform: str
    clip_index: int
    start: float
    end: float
    text: str
    score: float
    features: Dict[str, float]


def build_export_plan(
    selections: Dict[str, List[Dict[str, object]]],
    output_path: Path,
) -> List[ExportClip]:
    plan: List[ExportClip] = []
    for platform, clips in selections.items():
        for index, clip in enumerate(clips, start=1):
            plan.append(
                ExportClip(
                    platform=platform,
                    clip_index=index,
                    start=float(clip["start"]),
                    end=float(clip["end"]),
                    text=str(clip["text"]),
                    score=float(clip["score"]),
                    features=clip.get("features", {}),
                )
            )
    write_json(output_path, {"clips": [asdict(item) for item in plan]})
    return plan
