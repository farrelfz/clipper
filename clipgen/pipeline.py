from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from clipgen import asr, candidates, export, face, input as input_mod, probe, scoring, speaker, subtitles
from clipgen.config import Config, RenderConfig, resolve_platform_niche
from clipgen.logging import JsonlLogger
from clipgen.render import build_crop_path, extract_thumbnail, render_clip
from clipgen.transcode import mezzanine_transcode
from clipgen.utils import ensure_ffmpeg, read_json, run_command, write_json


class PipelineError(RuntimeError):
    pass


def analyze(source: str, config: Config, out_dir: Path, headers: List[str]) -> Dict[str, Path]:
    ensure_ffmpeg()
    logger = JsonlLogger(out_dir / "logs" / "pipeline.jsonl")
    with logger.timed("input", "resolve input"):
        resolved = input_mod.resolve_input(source, headers, out_dir / "cache")
    with logger.timed("probe", "ffprobe"):
        probe_path = out_dir / "analysis" / "probe.json"
        probe_data = probe.probe_video(resolved, probe_path)
        probe.write_probe_summary(probe_data, out_dir / "analysis" / "probe_summary.json")
    with logger.timed("transcode", "mezzanine transcode"):
        mezzanine_path = mezzanine_transcode(resolved, out_dir / "analysis")
    audio_path = out_dir / "analysis" / "audio.wav"
    with logger.timed("audio", "extract audio"):
        _extract_audio(mezzanine_path, audio_path)
    with logger.timed("vad", "voice activity detection"):
        vad_path = out_dir / "analysis" / "vad.json"
        vad_segments = _run_vad(audio_path, vad_path)
    with logger.timed("asr", "transcribe"):
        transcript_path = out_dir / "analysis" / "transcript.json"
        transcript = asr.transcribe(audio_path, transcript_path)
    with logger.timed("faces", "face detection"):
        tracks_path = out_dir / "analysis" / "tracks.json"
        detections = face.detect_faces(mezzanine_path, tracks_path, config.analysis.face_analysis_fps)
    with logger.timed("speaker", "speaker timeline"):
        timeline_path = out_dir / "analysis" / "speaker_timeline.json"
        speaker.build_speaker_timeline(
            detections=[
                {"time": det.time, "bbox": det.bbox, "score": det.score, "track_id": det.track_id}
                for det in detections
            ],
            vad_segments=[{"start": seg["start"], "end": seg["end"]} for seg in vad_segments],
            output_path=timeline_path,
            chunk_size=0.5,
            switch_hysteresis_s=config.analysis.switch_hysteresis_ms / 1000,
            confidence_threshold=config.analysis.confidence_threshold,
        )
    with logger.timed("candidates", "generate candidates"):
        candidates_path = out_dir / "analysis" / "candidates.json"
        candidates.build_candidates(
            transcript=asr.transcript_to_dict(transcript),
            config=config.clip_selection,
            output_path=candidates_path,
        )
    return {
        "mezzanine": mezzanine_path,
        "probe": probe_path,
        "transcript": transcript_path,
        "tracks": tracks_path,
        "speaker": timeline_path,
        "candidates": candidates_path,
        "vad": vad_path,
    }


def render(
    source: str,
    config: Config,
    out_dir: Path,
    headers: List[str],
    bundle_path: Path | None = None,
) -> None:
    analysis_paths = analyze(source, config, out_dir, headers)
    transcript = read_json(analysis_paths["transcript"])
    candidates_data = read_json(analysis_paths["candidates"])["candidates"][: config.bundle.candidate_pool]
    scored_per_platform: Dict[str, List[scoring.ScoredCandidate]] = {}
    for platform in config.bundle.platforms:
        niche = resolve_platform_niche(config, platform)
        if niche is None:
            raise PipelineError(f"Missing niche config for {platform}")
        scored_per_platform[platform] = scoring.score_candidates(
            candidates=candidates_data,
            config=config.clip_selection,
            niche=niche,
            platform=platform,
        )
    selected = scoring.select_non_overlapping(
        scored={platform: items for platform, items in scored_per_platform.items()},
        overlap_max_ratio=config.bundle.overlap_max_ratio,
        clips_per_platform=config.bundle.clips_per_platform,
    )
    export_plan_path = out_dir / "analysis" / "export_plan.json"
    plan_items = export.build_export_plan(
        selections={
            platform: [
                {
                    "start": item.start,
                    "end": item.end,
                    "text": item.text,
                    "score": item.score,
                    "features": item.features,
                }
                for item in items
            ]
            for platform, items in selected.items()
        },
        output_path=export_plan_path,
    )
    write_json(
        out_dir / "analysis" / "export.json",
        {"clips": [item.__dict__ for item in plan_items]},
    )
    render_config = _resolve_render_config(out_dir, config)
    _render_from_plan(
        analysis_paths=analysis_paths,
        config=config,
        render_config=render_config,
        out_dir=out_dir,
        plan=read_json(export_plan_path)["clips"],
        transcript=transcript,
    )
    if bundle_path:
        bundle_path.write_text(export_plan_path.read_text())


def render_from_plan(
    source: str,
    config: Config,
    out_dir: Path,
    headers: List[str],
    plan_path: Path,
) -> None:
    analysis_paths = _ensure_analysis(source, config, out_dir, headers)
    transcript = read_json(analysis_paths["transcript"])
    plan = read_json(plan_path)["clips"]
    render_config = _resolve_render_config(out_dir, config)
    _render_from_plan(analysis_paths, config, render_config, out_dir, plan, transcript)


def _ensure_analysis(
    source: str,
    config: Config,
    out_dir: Path,
    headers: List[str],
) -> Dict[str, Path]:
    analysis_dir = out_dir / "analysis"
    required = {
        "mezzanine": analysis_dir / "mezzanine.mp4",
        "transcript": analysis_dir / "transcript.json",
        "tracks": analysis_dir / "tracks.json",
        "speaker": analysis_dir / "speaker_timeline.json",
    }
    if all(path.exists() for path in required.values()):
        return required
    return analyze(source, config, out_dir, headers)


def _render_from_plan(
    analysis_paths: Dict[str, Path],
    config: Config,
    render_config: RenderConfig,
    out_dir: Path,
    plan: List[Dict[str, object]],
    transcript: Dict[str, object],
) -> None:
    detections = read_json(analysis_paths["tracks"])["detections"]
    timeline = read_json(analysis_paths["speaker"])["timeline"]
    for item in plan:
        platform = str(item["platform"])
        template = getattr(config.templates, platform)
        niche = resolve_platform_niche(config, platform)
        platform_dir = out_dir / "final" / platform
        captions_dir = platform_dir / "captions"
        thumbs_dir = platform_dir / "thumbs"
        meta_dir = platform_dir / "meta"
        clip_index = int(item["clip_index"])
        clip_name = f"clip_{clip_index:02d}"
        clip_start = float(item["start"])
        clip_end = float(item["end"])
        clip_path = platform_dir / f"{clip_name}.mp4"
        subtitles_paths = subtitles.generate_subtitles(
            transcript=transcript,
            clip_start=clip_start,
            clip_end=clip_end,
            output_dir=captions_dir / clip_name,
            config=config.subtitles,
            template=template,
            niche=niche,
        )
        pan_speed, zoom_rate = _platform_reframe_params(platform, config)
        crop_path = build_crop_path(
            detections=detections,
            timeline=timeline,
            clip_start=clip_start,
            clip_end=clip_end,
            width=config.render.width,
            height=config.render.height,
            safe_margin=config.analysis.crop_safe_margin_ratio,
            max_pan_speed=pan_speed,
            max_zoom_rate=zoom_rate,
        )
        render_clip(
            source=analysis_paths["mezzanine"],
            clip_start=clip_start,
            clip_end=clip_end,
            output_path=clip_path,
            crop_path=crop_path,
            config=render_config,
            subtitle_ass=subtitles_paths["ass"] if config.subtitles.burn_in else None,
        )
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumbs_dir / f"{clip_name}.jpg"
        extract_thumbnail(analysis_paths["mezzanine"], (clip_start + clip_end) / 2, thumb_path)
        meta_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            meta_dir / f"{clip_name}.json",
            {
                "platform": platform,
                "clip_index": clip_index,
                "start": clip_start,
                "end": clip_end,
                "text": item.get("text", ""),
                "score": item.get("score", 0),
                "features": item.get("features", {}),
                "subtitles": {
                    "srt": str(subtitles_paths["srt"].relative_to(out_dir)),
                    "vtt": str(subtitles_paths["vtt"].relative_to(out_dir)),
                    "ass": str(subtitles_paths["ass"].relative_to(out_dir)),
                },
                "thumbnail": str(thumb_path.relative_to(out_dir)),
            },
        )


def _extract_audio(source: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    run_command(cmd)


def _resolve_render_config(out_dir: Path, config: Config):
    summary_path = out_dir / "analysis" / "probe_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        fps = summary.get("video", {}).get("fps", config.render.fps_target)
        target = probe.determine_target_fps(float(fps), config.render.fps_target, config.render.fps_fallback)
        return config.render.model_copy(update={"fps_target": target})
    return config.render


def _platform_reframe_params(platform: str, config: Config) -> tuple[float, float]:
    if platform == "tiktok":
        return config.analysis.max_pan_speed * 1.1, config.analysis.max_zoom_rate * 1.1
    if platform == "reels":
        return config.analysis.max_pan_speed * 0.8, config.analysis.max_zoom_rate * 0.8
    return config.analysis.max_pan_speed, config.analysis.max_zoom_rate


def _run_vad(audio_path: Path, output_path: Path):
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    speech = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    segments = [
        {"start": entry["start"] / sample_rate, "end": entry["end"] / sample_rate}
        for entry in speech
    ]
    write_json(output_path, {"segments": segments})
    return segments
