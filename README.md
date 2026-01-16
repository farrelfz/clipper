# clipgen

Clipgen is a CLI tool that turns one long-form video into nine platform-ready short clips (3 each for TikTok, YouTube Shorts, Instagram Reels). It performs analysis (ASR, VAD, face tracking, speaker inference), produces non-overlapping highlight selections, and renders 9:16 outputs with professional subtitles.

## Requirements

- Python 3.11+
- ffmpeg + ffprobe on PATH
- Optional GPU support for faster ASR/vision

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### All-in-one

```bash
clipgen all /path/to/video.mp4 --config configs/bundle_3x3_high_quality.yaml --out out
```

### Analyze then render

```bash
clipgen analyze /path/to/video.mp4 --config configs/bundle_3x3_high_quality.yaml --out out
clipgen render /path/to/video.mp4 --bundle out/analysis/export_plan.json --config configs/bundle_3x3_high_quality.yaml --out out
```

### Direct-download URLs

```bash
clipgen all "https://example.com/video.mp4" --config configs/bundle_3x3_high_quality.yaml --out out
clipgen all "https://example.com/video.mp4" --header "Authorization: Bearer $TOKEN" --config configs/bundle_3x3_high_quality.yaml --out out
```

> **Compliance note:** Only direct-download video URLs are accepted (video content-type or byte-range support). clipgen **does not** download streaming pages, DRM-protected media, or URLs behind login walls. Provide a local file or a direct-download link that you are authorized to use.

## Output Structure

```
OUTDIR/
  final/
    tiktok/
      clip_01.mp4 clip_02.mp4 clip_03.mp4
      captions/ clip_01.srt clip_01.vtt clip_01.ass ...
      thumbs/ clip_01.jpg ...
      meta/ clip_01.json ...
    shorts/ (same)
    reels/ (same)
  analysis/
    probe.json
    transcript.json
    tracks.json
    speaker_timeline.json
    candidates.json
    export_plan.json
    export.json
  logs/
    pipeline.jsonl
```

## Configuration

A high-quality 3x3 bundle is included at:

```
configs/bundle_3x3_high_quality.yaml
```

This config uses:
- libx264 preset `medium`, CRF 17
- 1080x1920 output
- 60 fps target / 30 fps fallback
- loudness normalization
- platform-specific subtitle templates and niche lexicons

## JSON Schemas

- `schemas/export_plan.schema.json`
- `schemas/clip_meta.schema.json`

## Troubleshooting

- **ffmpeg missing**: Install ffmpeg and ensure `ffmpeg` and `ffprobe` are on PATH.
- **ASR model sizes**: faster-whisper uses local models; large models provide better accuracy at higher compute cost.
- **MediaPipe install**: On some systems, `mediapipe` requires extra system dependencies; see MediaPipe docs.
- **Performance**: Face detection and ASR are the slowest stages. Use GPU (`--device cuda`) if available.
