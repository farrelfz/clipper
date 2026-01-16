from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .asr import Segment
from .config import SubtitleTemplate


@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float, sep: str = ",") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", sep)


def line_break(text: str, max_chars: int, max_lines: int) -> str:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        if len(" ".join(current + [word])) <= max_chars:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines - 1:
                break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    return "\n".join(lines)


def highlight_keywords(text: str, keywords: Iterable[str], color: str) -> str:
    highlighted = text
    for keyword in keywords:
        if not keyword:
            continue
        highlighted = highlighted.replace(
            keyword,
            f"{{\\c{color}}}{keyword}{{\\c&HFFFFFF&}}",
        )
    return highlighted


def build_subtitle_lines(segments: List[Segment], max_chars: int, max_lines: int) -> List[SubtitleLine]:
    lines = []
    for seg in segments:
        text = line_break(seg.text, max_chars=max_chars, max_lines=max_lines)
        lines.append(SubtitleLine(start=seg.start, end=seg.end, text=text))
    return lines


def write_srt(lines: List[SubtitleLine], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, line in enumerate(lines, start=1):
            handle.write(f"{idx}\n")
            handle.write(f"{format_timestamp(line.start)} --> {format_timestamp(line.end)}\n")
            handle.write(f"{line.text}\n\n")


def write_vtt(lines: List[SubtitleLine], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for line in lines:
            handle.write(f"{format_timestamp(line.start, sep='.') } --> {format_timestamp(line.end, sep='.') }\n")
            handle.write(f"{line.text}\n\n")


def ass_header(template: SubtitleTemplate, safe_margin_ratio: float) -> str:
    margin_v = int(1080 * safe_margin_ratio)
    style = (
        f"Style: Default,{template.font},{template.font_size},"
        f"{template.primary_color},{template.outline_color},&H00000000,&H64000000,"
        f"{1 if template.bold else 0},0,0,0,100,100,0,0,1,"
        f"{template.outline},{template.shadow},2,{margin_v},{margin_v},30,1"
    )
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,"
        "Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"{style}\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    return header


def write_ass(
    lines: List[SubtitleLine],
    path: Path,
    template: SubtitleTemplate,
    safe_margin_ratio: float,
    highlight_terms: Iterable[str],
) -> None:
    header = ass_header(template, safe_margin_ratio)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for line in lines:
            text = line.text.replace("\n", "\\N")
            if template.name == "tiktok_bold_highlight":
                text = highlight_keywords(text, highlight_terms, "&H00FFFF&")
            elif template.name == "shorts_clean":
                text = highlight_keywords(text, highlight_terms, "&H00FF00&")
            elif template.name == "reels_elegant":
                text = highlight_keywords(text, highlight_terms, "&HFFD700&")
            handle.write(
                f"Dialogue: 0,{format_timestamp(line.start, sep='.') },{format_timestamp(line.end, sep='.') },"
                f"Default,,0,0,0,,{text}\n"
            )
