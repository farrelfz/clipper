from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class BundleConfig(BaseModel):
    platforms: List[str] = Field(default_factory=lambda: ["tiktok", "shorts", "reels"])
    clips_per_platform: int = 3
    overlap_max_ratio: float = 0.25
    candidate_pool: int = 80


class AnalysisConfig(BaseModel):
    speaker_mode: str = "hybrid"
    face_backend: str = "mediapipe"
    face_analysis_fps: float = 12.0
    switch_hysteresis_ms: int = 400
    confidence_threshold: float = 0.12
    max_pan_speed: float = 0.25
    max_zoom_rate: float = 0.08
    crop_safe_margin_ratio: float = 0.12


class DurationRange(BaseModel):
    min_seconds: float
    max_seconds: float


class ClipSelectionConfig(BaseModel):
    require_hook_first_seconds: float = 2.5
    avoid_mid_sentence_cut: bool = True
    use_scene_cut_hint: bool = True
    tiktok: DurationRange = DurationRange(min_seconds=12, max_seconds=35)
    shorts: DurationRange = DurationRange(min_seconds=15, max_seconds=45)
    reels: DurationRange = DurationRange(min_seconds=20, max_seconds=60)


class RenderAudioConfig(BaseModel):
    codec: str = "aac"
    sample_rate: int = 48000
    bitrate: str = "192k"
    loudness_normalize: bool = True


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
    sharpen: bool = False
    denoise: bool = False
    audio: RenderAudioConfig = RenderAudioConfig()


class SubtitleConfig(BaseModel):
    punctuation: bool = True
    max_lines: int = 2
    max_chars_per_line: int = 26
    highlight: bool = True
    burn_in: bool = True
    safe_margin_ratio: float = 0.08


class SubtitleTemplate(BaseModel):
    name: str
    font: str = "Arial"
    font_size: int = 60
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000FFEE"
    outline_color: str = "&H00000000"
    outline: int = 2
    shadow: int = 0
    box: bool = False
    box_color: str = "&H55000000"


class TemplatesConfig(BaseModel):
    tiktok: SubtitleTemplate
    shorts: SubtitleTemplate
    reels: SubtitleTemplate


class NicheConfig(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    cta_terms: List[str] = Field(default_factory=list)


class NicheBundleConfig(BaseModel):
    tiktok: NicheConfig
    shorts: NicheConfig
    reels: NicheConfig


class Config(BaseModel):
    bundle: BundleConfig = BundleConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    clip_selection: ClipSelectionConfig = ClipSelectionConfig()
    render: RenderConfig = RenderConfig()
    subtitles: SubtitleConfig = SubtitleConfig()
    templates: TemplatesConfig
    niches: NicheBundleConfig


class PlatformWeights(BaseModel):
    hook_strength: float
    keyword_hits: float
    audio_energy_peaks: float
    speaking_rate: float
    structure: float
    novelty: float
    penalty: float


PLATFORM_WEIGHTS: Dict[str, PlatformWeights] = {
    "tiktok": PlatformWeights(
        hook_strength=0.28,
        keyword_hits=0.18,
        audio_energy_peaks=0.18,
        speaking_rate=0.14,
        structure=0.08,
        novelty=0.08,
        penalty=0.06,
    ),
    "shorts": PlatformWeights(
        hook_strength=0.16,
        keyword_hits=0.16,
        audio_energy_peaks=0.12,
        speaking_rate=0.16,
        structure=0.2,
        novelty=0.12,
        penalty=0.08,
    ),
    "reels": PlatformWeights(
        hook_strength=0.14,
        keyword_hits=0.14,
        audio_energy_peaks=0.12,
        speaking_rate=0.12,
        structure=0.24,
        novelty=0.16,
        penalty=0.08,
    ),
}


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)


def resolve_platform_niche(config: Config, platform: str) -> Optional[NicheConfig]:
    return getattr(config.niches, platform, None)
