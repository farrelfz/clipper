from __future__ import annotations

from pathlib import Path
from typing import Optional

from clipgen.utils import run_command


def mezzanine_transcode(source: Path, output_dir: Path, overwrite: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "mezzanine.mp4"
    if target.exists() and not overwrite:
        return target
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        str(target),
    ]
    run_command(cmd)
    return target
