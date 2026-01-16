from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from clipgen.config import NicheConfig, SubtitleConfig, SubtitleTemplate


@dataclass
class CaptionLine:
    start: float
    end: float
    text: str


def generate_subtitles(
    transcript: Dict[str, object],
    clip_start: float,
    clip_end: float,
    output_dir: Path,
    config: SubtitleConfig,
    template: SubtitleTemplate,
    niche: NicheConfig,
) -> Dict[str, Path]:
    words = _collect_words(transcript, clip_start, clip_end)
    captions = _build_caption_lines(words, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = output_dir / "captions.srt"
    vtt_path = output_dir / "captions.vtt"
    ass_path = output_dir / "captions.ass"
    srt_path.write_text(_to_srt(captions))
    vtt_path.write_text(_to_vtt(captions))
    ass_path.write_text(_to_ass(captions, template, niche, config))
    return {"srt": srt_path, "vtt": vtt_path, "ass": ass_path}


def _collect_words(
    transcript: Dict[str, object],
    clip_start: float,
    clip_end: float,
) -> List[Dict[str, object]]:
    words: List[Dict[str, object]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start = float(word["start"])
            end = float(word["end"])
            if end < clip_start or start > clip_end:
                continue
            words.append({"start": start, "end": end, "word": word["word"].strip()})
    return words


def _build_caption_lines(words: List[Dict[str, object]], config: SubtitleConfig) -> List[CaptionLine]:
    if not words:
        return []
    captions: List[CaptionLine] = []
    current_lines: List[str] = []
    current_line = ""
    current_start = float(words[0]["start"])
    current_end = float(words[0]["end"])
    for word in words:
        token = word["word"]
        candidate = f"{current_line} {token}".strip() if current_line else token
        current_end = float(word["end"])
        if len(candidate) > config.max_chars_per_line and current_line:
            current_lines.append(current_line)
            current_line = token
            if len(current_lines) >= config.max_lines:
                captions.append(
                    CaptionLine(
                        start=current_start,
                        end=current_end,
                        text="\\N".join(current_lines),
                    )
                )
                current_lines = []
                current_start = float(word["start"])
        else:
            current_line = candidate
    if current_line:
        current_lines.append(current_line)
    if current_lines:
        captions.append(
            CaptionLine(
                start=current_start,
                end=current_end,
                text="\\N".join(current_lines),
            )
        )
    return captions




def _to_srt(lines: Iterable[CaptionLine]) -> str:
    output = []
    for index, line in enumerate(lines, start=1):
        output.append(str(index))
        output.append(f"{_format_time(line.start)} --> {_format_time(line.end)}")
        output.append(line.text.replace("\\N", "\n"))
        output.append("")
    return "\n".join(output)


def _to_vtt(lines: Iterable[CaptionLine]) -> str:
    output = ["WEBVTT", ""]
    for line in lines:
        output.append(f"{_format_time(line.start)} --> {_format_time(line.end)}")
        output.append(line.text.replace("\\N", "\n"))
        output.append("")
    return "\n".join(output)


def _to_ass(
    lines: Iterable[CaptionLine],
    template: SubtitleTemplate,
    niche: NicheConfig,
    config: SubtitleConfig,
) -> str:
    highlight_terms = {kw.lower() for kw in niche.keywords + niche.cta_terms}
    style = (
        f"Style: Default,{template.font},{template.font_size},"
        f"{template.primary_color},{template.primary_color},{template.outline_color},"
        f"{template.outline_color},{template.outline},{template.shadow},"
        f"{1 if template.box else 0},{template.box_color},"
        f"0,0,0,0,100,100,0,0,1,2,2,{int(config.safe_margin_ratio * 100)},"
        f"{int(config.safe_margin_ratio * 100)},50,1"
    )
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        style,
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = []
    for line in lines:
        if config.highlight:
            text = _apply_highlight(line.text, highlight_terms, template.highlight_color)
        else:
            text = line.text
        body.append(
            f"Dialogue: 0,{_format_ass_time(line.start)},{_format_ass_time(line.end)},"
            f"Default,,0,0,0,,{text}"
        )
    return "\n".join(header + body)


def _apply_highlight(text: str, keywords: set[str], color: str) -> str:
    output_words = []
    for word in text.split():
        token = word.strip().lower().strip(",.!?")
        if token in keywords:
            output_words.append(f"{{\\c{color}}}{word}{{\\c}}")
        else:
            output_words.append(word)
    return " ".join(output_words)


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def _format_ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:01d}:{minutes:02d}:{secs:05.2f}"
