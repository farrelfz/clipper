from clipgen.candidates import Candidate
from clipgen.scoring import score_candidates


def test_scoring_applies_weights():
    candidates = [
        Candidate(start=0, end=10, text="Big 10X growth story"),
        Candidate(start=12, end=20, text="simple explanation"),
    ]
    scored = score_candidates(candidates, keywords=["growth"], platform="tiktok")
    assert scored[0].score >= scored[1].score
