from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console

from .asr import Segment, transcribe_audio
from .candidates import Candidate, generate_candidates
from .config import AppConfig, load_config
from .export import export_candidates, export_plan, export_speaker_timeline, export_tracks, write_json
from .face import detect_faces
from .input import InputError, resolve_input
from .logging import JsonlLogger
from .probe import estimate_fps, pick_video_stream, run_ffprobe, select_target_fps
from .render import (
    _limit_speed,
    _smooth_keyframes,
    _target_crop_size,
    build_crop_filter,
    build_keyframes,
    render_clip,
    render_thumbnail,
)
from .scoring import build_export_plan
from .speaker import SpeakerChunk, infer_speaker_timeline
from .subtitles import build_subtitle_lines, write_ass, write_srt, write_vtt
from .vad import run_vad

app = typer.Typer(add_completion=False)
console = Console()


def _ensure_dirs(outdir: Path) -> Dict[str, Path]:
    dirs = {
        "final": outdir / "final",
        "analysis": outdir / "analysis",
        "logs": outdir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_transcript(segments: List[Segment], path: Path) -> None:
    payload = [
        {
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
            "words": [asdict(word) for word in seg.words],
        }
        for seg in segments
    ]
    write_json(path, payload)


def analyze_pipeline(
    input_path: str,
    config: AppConfig,
    outdir: Path,
    headers: Optional[List[str]],
    device: str,
) -> Dict[str, Path]:
    dirs = _ensure_dirs(outdir)
    logger = JsonlLogger(dirs["logs"] / "pipeline.jsonl")

    header_map: Dict[str, str] = {}
    if headers:
        for header in headers:
            if ":" not in header:
                raise InputError("Headers must be in 'Key: Value' format.")
            key, value = header.split(":", 1)
            header_map[key.strip()] = value.strip()

    source = resolve_input(input_path, header_map, logger)
    probe = run_ffprobe(source)
    write_json(dirs["analysis"] / "probe.json", probe)
    video_stream = pick_video_stream(probe)

    segments = transcribe_audio(source, model_size="medium", logger=logger)
    _write_transcript(segments, dirs["analysis"] / "transcript.json")

    speech_windows = run_vad(source, logger)
    write_json(dirs["analysis"] / "vad.json", [asdict(win) for win in speech_windows])

    track_result = detect_faces(source, config.analysis.face_analysis_fps, logger)
    export_tracks(track_result, dirs["analysis"] / "tracks.json")

    timeline = infer_speaker_timeline(
        source,
        track_result,
        speech_windows,
        chunk_size=0.5,
        hysteresis_ms=config.analysis.switch_hysteresis_ms,
        confidence_threshold=config.analysis.confidence_threshold,
        logger=logger,
    )
    export_speaker_timeline(timeline, dirs["analysis"] / "speaker_timeline.json")

    min_s = min(
        config.clip_selection.tiktok.min_s,
        config.clip_selection.shorts.min_s,
        config.clip_selection.reels.min_s,
    )
    max_s = max(
        config.clip_selection.tiktok.max_s,
        config.clip_selection.shorts.max_s,
        config.clip_selection.reels.max_s,
    )
    candidates = generate_candidates(
        segments,
        speech_windows,
        min_s=min_s,
        max_s=max_s,
        candidate_pool=config.bundle.candidate_pool,
    )
    export_candidates(candidates, dirs["analysis"] / "candidates.json")

    keywords = {
        "tiktok": config.niches.tiktok.keywords + config.niches.tiktok.cta,
        "shorts": config.niches.shorts.keywords + config.niches.shorts.cta,
        "reels": config.niches.reels.keywords + config.niches.reels.cta,
    }
    plan = build_export_plan(
        candidates,
        keywords=keywords,
        overlap_max_ratio=config.bundle.overlap_max_ratio,
        clips_per_platform=config.bundle.clips_per_platform,
    )
    export_payload = {
        "platforms": {
            platform: [
                {
                    "start": item.candidate.start,
                    "end": item.candidate.end,
                    "text": item.candidate.text,
                    "score": item.score,
                    "features": item.features,
                }
                for item in items
            ]
            for platform, items in plan.items()
        }
    }
    export_plan(export_payload, dirs["analysis"] / "export_plan.json")
    write_json(dirs["analysis"] / "export.json", export_payload)

    return dirs


@app.command()
def analyze(
    input_path: str = typer.Argument(..., help="Input file path or direct-download URL."),
    config_path: Path = typer.Option(..., "--config", help="Path to YAML config."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    header: Optional[List[str]] = typer.Option(None, "--header", help="Custom header 'Key: Value'."),
    device: str = typer.Option("cpu", "--device", help="Device (cpu/cuda)."),
) -> None:
    config = load_config(config_path)
    analyze_pipeline(input_path, config, out, header, device)
    console.print("Analysis complete.")


@app.command()
def render(
    input_path: str = typer.Argument(..., help="Input file path or direct-download URL."),
    bundle: Path = typer.Option(..., "--bundle", help="Path to analysis/export_plan.json"),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    config_path: Path = typer.Option(..., "--config", help="Path to YAML config."),
    header: Optional[List[str]] = typer.Option(None, "--header", help="Custom header 'Key: Value'."),
    device: str = typer.Option("cpu", "--device", help="Device (cpu/cuda)."),
) -> None:
    config = load_config(config_path)
    dirs = _ensure_dirs(out)
    logger = JsonlLogger(dirs["logs"] / "pipeline.jsonl")

    header_map: Dict[str, str] = {}
    if header:
        for header_item in header:
            key, value = header_item.split(":", 1)
            header_map[key.strip()] = value.strip()

    source = resolve_input(input_path, header_map, logger)
    probe = run_ffprobe(source)
    video_stream = pick_video_stream(probe)
    source_fps = estimate_fps(video_stream)
    target_fps = select_target_fps(source_fps)

    plan = json.loads(bundle.read_text())
    segments = transcribe_audio(source, model_size="medium", logger=logger)

    track_path = out / "analysis" / "tracks.json"
    speaker_path = out / "analysis" / "speaker_timeline.json"
    if not track_path.exists() or not speaker_path.exists():
        raise RuntimeError("Missing analysis outputs. Run analyze first.")

    track_data = json.loads(track_path.read_text())
    speaker_data = json.loads(speaker_path.read_text())

    source_w = int(video_stream.get("width", config.render.width))
    source_h = int(video_stream.get("height", config.render.height))
    crop_w, crop_h = _target_crop_size(source_w, source_h)

    track_centers = []
    for track in track_data["tracks"]:
        detections = track["detections"]
        centers = []
        for bbox in detections.values():
            x1, y1, x2, y2 = bbox
            centers.append(((x1 + x2) / 2 * source_w, (y1 + y2) / 2 * source_h))
        if centers:
            avg_x = sum(c[0] for c in centers) / len(centers)
            avg_y = sum(c[1] for c in centers) / len(centers)
            track_centers.append((avg_x, avg_y))
        else:
            track_centers.append((source_w / 2, source_h / 2))

    speaker_chunks = [SpeakerChunk(**chunk) for chunk in speaker_data]

    for platform, clips in plan["platforms"].items():
        platform_dir = dirs["final"] / platform
        captions_dir = platform_dir / "captions"
        thumbs_dir = platform_dir / "thumbs"
        meta_dir = platform_dir / "meta"
        for idx, clip in enumerate(clips, start=1):
            clip_start = clip["start"]
            clip_end = clip["end"]
            clip_segments = [seg for seg in segments if seg.start >= clip_start and seg.end <= clip_end]
            subtitle_lines = build_subtitle_lines(
                clip_segments,
                max_chars=config.subtitles.max_chars_per_line,
                max_lines=config.subtitles.max_lines,
            )
            template = getattr(config.templates, platform)
            highlight_terms = getattr(config.niches, platform).keywords
            ass_path = captions_dir / f"clip_{idx:02d}.ass"
            write_ass(
                subtitle_lines,
                ass_path,
                template=template,
                safe_margin_ratio=config.subtitles.safe_margin_ratio,
                highlight_terms=highlight_terms,
            )
            write_srt(subtitle_lines, captions_dir / f"clip_{idx:02d}.srt")
            write_vtt(subtitle_lines, captions_dir / f"clip_{idx:02d}.vtt")

            chunk_window = [
                chunk
                for chunk in speaker_chunks
                if chunk.start >= clip_start and chunk.end <= clip_end
            ]
            keyframes = build_keyframes(
                chunk_window,
                track_centers=track_centers,
                default_center=(source_w / 2, source_h / 2),
            )
            speed = config.analysis.max_pan_speed
            if platform == "tiktok":
                speed *= 1.2
            elif platform == "reels":
                speed *= 0.8
            keyframes = _limit_speed(keyframes, max_delta=speed * source_w)
            keyframes = _smooth_keyframes(keyframes, alpha=0.3)

            crop_filter = build_crop_filter(keyframes, crop_w, crop_h, source_w, source_h)

            out_path = platform_dir / f"clip_{idx:02d}.mp4"
            render_clip(
                input_path=source,
                output_path=out_path,
                subtitle_ass=ass_path if config.subtitles.burn_in else None,
                crop_filter=crop_filter,
                render_config=config.render,
                target_fps=target_fps,
                start=clip_start,
                end=clip_end,
                logger=logger,
            )
            render_thumbnail(source, thumbs_dir / f"clip_{idx:02d}.jpg", (clip_start + clip_end) / 2)
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "platform": platform,
                "start": clip_start,
                "end": clip_end,
                "score": clip["score"],
                "features": clip["features"],
            }
            write_json(meta_dir / f"clip_{idx:02d}.json", meta)

    console.print("Render complete.")


@app.command()
def all(
    input_path: str = typer.Argument(..., help="Input file path or direct-download URL."),
    config_path: Path = typer.Option(..., "--config", help="Path to YAML config."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    header: Optional[List[str]] = typer.Option(None, "--header", help="Custom header 'Key: Value'."),
    device: str = typer.Option("cpu", "--device", help="Device (cpu/cuda)."),
) -> None:
    config = load_config(config_path)
    analyze_pipeline(input_path, config, out, header, device)
    export_plan_path = out / "analysis" / "export_plan.json"
    render(
        input_path=input_path,
        bundle=export_plan_path,
        out=out,
        config_path=config_path,
        header=header,
        device=device,
    )
