from clipgen.scoring import ScoredCandidate, select_non_overlapping


def test_anti_overlap_selection():
    a = ScoredCandidate(start=0, end=10, text="A", score=10, features={})
    b = ScoredCandidate(start=5, end=15, text="B", score=9, features={})
    c = ScoredCandidate(start=20, end=30, text="C", score=8, features={})
    selected = select_non_overlapping(
        scored={"tiktok": [a, b, c]}, overlap_max_ratio=0.25, clips_per_platform=2
    )
    assert selected["tiktok"][0].text == "A"
    assert selected["tiktok"][1].text == "C"
