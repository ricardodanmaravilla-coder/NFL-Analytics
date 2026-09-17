import cloudrun_api as api


def test_spread_is_always_lean_with_zero_stake():
    pick = api.market_candidate(
        "A @ B", "B -3", "SPREAD", -3.0,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "LEAN"
    assert pick["stake"] == 0.0


def test_total_keeps_auto_bet_when_filters_pass():
    pick = api.market_candidate(
        "A @ B", "Over 45.5", "TOTAL", 45.5,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "BET"
    assert pick["stake"] > 0.0


def test_health_exposes_production_guards():
    h = api.health()
    assert h["version"] == "3.9"
    assert h["spread_auto_bet"] is False
    assert h["pbp_asof_cutoff"] is True
