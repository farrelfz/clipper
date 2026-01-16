from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class BundleConfig(BaseModel):
    platforms: List[Literal["tiktok", "shorts", "reels"]] = Field(
        default_factory=lambda: ["tiktok", "shorts", "reels"]
    )
    clips_per_platform: int = 3
    overlap_max_ratio: float = 0.25
    candidate_pool: int = 80


class AnalysisConfig(BaseModel):
    speaker_mode: Literal["hybrid"] = "hybrid"
    face_backend: Literal["mediapipe"] = "mediapipe"
    face_analysis_fps: float = 12.0
    switch_hysteresis_ms: int = 500
    confidence_threshold: float = 0.15
    max_pan_speed: float = 0.18
    max_zoom_rate: float = 0.15
    crop_safe_margin_ratio: float = 0.15


class PlatformDurations(BaseModel):
    min_s: float
    max_s: float


class ClipSelectionConfig(BaseModel):
    require_hook_first_seconds: float = 3.0
    avoid_mid_sentence_cut: bool = True
    use_scene_cut_hint: bool = True
    tiktok: PlatformDurations
    shorts: PlatformDurations
    reels: PlatformDurations


class RenderConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    fps_target: int = 60
    fps_fallback: int = 30
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 17
    scaler: str = "lanczos"
    pix_fmt: str = "yuv420p"
    audio_codec: str = "aac"
    audio_rate: int = 48000
    audio_bitrate: str = "192k"
    loudness_normalize: bool = True
    sharpen: bool = False
    denoise: bool = False


class SubtitleConfig(BaseModel):
    punctuation: bool = True
    max_lines: int = 2
    max_chars_per_line: int = 26
    highlight: bool = True
    burn_in: bool = True
    safe_margin_ratio: float = 0.08


class SubtitleTemplate(BaseModel):
    name: str
    font: str
    font_size: int
    primary_color: str
    outline_color: str
    outline: int
    shadow: int
    bold: bool
    box: bool
    box_color: Optional[str] = None


class TemplateConfig(BaseModel):
    tiktok: SubtitleTemplate
    shorts: SubtitleTemplate
    reels: SubtitleTemplate


class NicheConfig(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    cta: List[str] = Field(default_factory=list)


class NichesConfig(BaseModel):
    tiktok: NicheConfig
    shorts: NicheConfig
    reels: NicheConfig


class AppConfig(BaseModel):
    bundle: BundleConfig
    analysis: AnalysisConfig
    clip_selection: ClipSelectionConfig
    render: RenderConfig
    subtitles: SubtitleConfig
    templates: TemplateConfig
    niches: NichesConfig


def load_config(path: Path) -> AppConfig:
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config not found: {path}") from exc

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config: {exc}") from exc


def save_config(config: AppConfig, path: Path) -> None:
    data = config.model_dump()
    path.write_text(yaml.safe_dump(data, sort_keys=False))
