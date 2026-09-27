import cloudrun_api as api


def test_spread_recommendation_stays_enabled_with_risk_cap():
    pick = api.market_candidate(
        "A @ B", "B -3", "SPREAD", -3.0,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "BET"
    assert 0.0 < pick["stake"] <= 250.0


def test_total_keeps_recommendation_when_filters_pass():
    pick = api.market_candidate(
        "A @ B", "Over 45.5", "TOTAL", 45.5,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "BET"
    assert 0.0 < pick["stake"] <= 250.0


def test_health_exposes_production_guards():
    h = api.health()
    assert h["version"] == "3.9"
    assert h["spread_auto_bet"] is True
    assert h["total_auto_bet"] is True
    assert h["pbp_asof_cutoff"] is True


def test_load_history_uses_fresh_validated_loader(monkeypatch):
    games = object()
    pbp = object()
    api.load_history.cache_clear()
    monkeypatch.setattr(api, "load_production_history", lambda prefer_remote=True: (games, pbp, "REMOTE_PARQUET"))
    got_games, got_pbp = api.load_history()
    assert got_games is games
    assert got_pbp is pbp
    api.load_history.cache_clear()


def test_moneyline_positive_odds_remain_non_auto_until_oos_gate_proves_value():
    pick = api.candidate(
        "A @ B", "B ML", 65.0, [64.0, 63.0],
        +120, -140, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "LEAN"
    assert pick["stake"] == 0.0
