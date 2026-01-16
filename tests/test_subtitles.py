from clipgen.subtitles import _build_caption_lines
from clipgen.config import SubtitleConfig


def test_line_breaking_limits_chars():
    words = [
        {"start": 0.0, "end": 0.5, "word": "This"},
        {"start": 0.5, "end": 1.0, "word": "is"},
        {"start": 1.0, "end": 1.5, "word": "a"},
        {"start": 1.5, "end": 2.0, "word": "longer"},
        {"start": 2.0, "end": 2.5, "word": "sentence"},
        {"start": 2.5, "end": 3.0, "word": "for"},
        {"start": 3.0, "end": 3.5, "word": "testing"},
    ]
    config = SubtitleConfig(max_chars_per_line=12)
    lines = _build_caption_lines(words, config)
    assert lines
    for line in lines:
        for part in line.text.split("\\N"):
            assert len(part) <= 12
