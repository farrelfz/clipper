# Clipgen

Clipgen is a CLI tool that turns a single long-form video into nine short-form clips (3 TikTok, 3 Shorts, 3 Reels) with platform-specific selection logic, speaker-aware reframing, and styled subtitles.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Prerequisites

- Python 3.11+
- FFmpeg + FFprobe (required)
- Optional GPU for faster ASR

## Usage

### Full pipeline

```bash
clipgen all /path/to/video.mp4 --config configs/bundle_3x3_high_quality.yaml --out /tmp/clipgen_output
```

### Direct-download URL (authorized)

```bash
clipgen all "https://cdn.example.com/video.mp4" --config configs/bundle_3x3_high_quality.yaml --out /tmp/clipgen_output \
  --header "Authorization: Bearer YOUR_TOKEN"
```

### Analyze only

```bash
clipgen analyze /path/to/video.mp4 --config configs/bundle_3x3_high_quality.yaml --out /tmp/clipgen_output
```

### Render using an existing export plan

```bash
clipgen render /path/to/video.mp4 \
  --bundle /tmp/clipgen_output/analysis/export_plan.json \
  --config configs/bundle_3x3_high_quality.yaml \
  --out /tmp/clipgen_output
```

## Output Structure

```
OUTDIR/
  final/
    tiktok/
      clip_01.mp4 clip_02.mp4 clip_03.mp4
      captions/ clip_01/captions.srt clip_01/captions.vtt clip_01/captions.ass ...
      thumbs/ clip_01.jpg ...
      meta/ clip_01.json ...
    shorts/ (same)
    reels/ (same)
  analysis/
    probe.json
    probe_summary.json
    transcript.json
    tracks.json
    speaker_timeline.json
    candidates.json
    export_plan.json
    vad.json
    audio.wav
    mezzanine.mp4
  logs/
    pipeline.jsonl
```

## JSON Schemas

- `schemas/export_plan.schema.json` documents the export plan format.
- `schemas/clip_meta.schema.json` documents the clip meta JSON outputs.

## Troubleshooting

- **ASR is slow**: Use a smaller faster-whisper model or run on GPU.
- **MediaPipe install issues**: Ensure you have compatible Python and system dependencies for OpenCV.
- **FFmpeg missing**: Install ffmpeg and ensure it is in your PATH.

## URL and Compliance Restrictions

Clipgen only accepts local file paths or direct-download HTTP/HTTPS video URLs. It does **not** download from video platforms that require login/DRM or HTML pages. If a URL is not a direct video file, provide a local file path or a direct-download URL that you are authorized to use.
