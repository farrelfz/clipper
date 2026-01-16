from clipgen.config import ClipSelectionConfig, NicheConfig
from clipgen.scoring import score_candidates


def test_platform_weight_application():
    candidates = [
        {"start": 0, "end": 20, "text": "This is a quick 123 hack", "tokens": ["This", "is", "a", "quick", "123", "hack"]},
        {"start": 30, "end": 50, "text": "Plain explanation without hook", "tokens": ["Plain", "explanation", "without", "hook"]},
    ]
    config = ClipSelectionConfig()
    niche = NicheConfig(name="edu", keywords=["hack"], cta_terms=[])
    scored = score_candidates(candidates, config, niche, platform="tiktok")
    assert scored[0].text == "This is a quick 123 hack"
