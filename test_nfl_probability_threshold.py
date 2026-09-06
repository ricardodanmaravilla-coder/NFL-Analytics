import cloudrun_api as api


def test_probability_threshold_is_56_percent():
    assert api.MIN_PROBABILITY == 56.0


def test_candidate_rejects_below_56_and_accepts_56_when_value_passes(monkeypatch):
    monkeypatch.setattr(api, "primary_with_agreement", lambda primary, support, max_disagreement=15.0: primary)

    below = api.candidate("A @ B", "B ML", 55.9, [55.9], 100, -130, 5000)
    assert below is None

    at_threshold = api.candidate("A @ B", "B ML", 56.0, [56.0], 100, -130, 5000)
    assert at_threshold is not None
    assert at_threshold["probability"] == 56.0
