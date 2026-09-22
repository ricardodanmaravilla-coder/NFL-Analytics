import cloudrun_api as api


def test_spread_recommendation_stays_enabled_with_risk_cap():
    pick = api.market_candidate(
        "A @ B", "B -3", "SPREAD", -3.0,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "BET"
    assert 0.0 < pick["stake"] <= 150.0


def test_total_keeps_recommendation_when_filters_pass():
    pick = api.market_candidate(
        "A @ B", "Over 45.5", "TOTAL", 45.5,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "BET"
    assert 0.0 < pick["stake"] <= 150.0


def test_health_exposes_production_guards():
    h = api.health()
    assert h["version"] == "3.9"
    assert h["spread_auto_bet"] is True
    assert h["total_auto_bet"] is True
    assert h["pbp_asof_cutoff"] is True
