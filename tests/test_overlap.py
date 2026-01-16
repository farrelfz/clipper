from clipgen.candidates import Candidate
from clipgen.scoring import CandidateScore, select_with_overlap_guard


def test_overlap_guard_blocks_overlap():
    base = Candidate(start=0, end=10, text="a")
    overlap = Candidate(start=2, end=8, text="b")
    ok = Candidate(start=12, end=20, text="c")
    ranked = [
        CandidateScore(candidate=base, score=1.0, features={}),
        CandidateScore(candidate=overlap, score=0.9, features={}),
        CandidateScore(candidate=ok, score=0.8, features={}),
    ]
    selected = select_with_overlap_guard(ranked, existing=[], overlap_max_ratio=0.25, limit=2)
    assert [item.candidate for item in selected] == [base, ok]
