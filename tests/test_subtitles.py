from clipgen.subtitles import line_break


def test_line_break_respects_limits():
    text = "This is a sentence that should wrap nicely into two lines"
    result = line_break(text, max_chars=20, max_lines=2)
    lines = result.split("\n")
    assert len(lines) <= 2
    assert all(len(line) <= 20 for line in lines)
