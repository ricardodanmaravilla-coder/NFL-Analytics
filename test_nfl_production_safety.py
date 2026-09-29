import cloudrun_api as api


def test_spread_recommendation_stays_enabled_with_risk_cap():
    pick = api.market_candidate(
        "A @ B", "B -3", "SPREAD", -3.0,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "LEAN"
    assert pick["stake"] == 0.0


def test_total_keeps_recommendation_when_filters_pass():
    pick = api.market_candidate(
        "A @ B", "Over 45.5", "TOTAL", 45.5,
        65.0, 64.0, -110, -110, bankroll=5000,
    )
    assert pick is not None
    assert pick["action"] == "LEAN"
    assert pick["stake"] == 0.0


def test_health_exposes_production_guards():
    h = api.health()
    assert h["version"] == "3.9"
    assert h["spread_auto_bet"] is False
    assert h["total_auto_bet"] is False
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


def test_moneyline_positive_odds_outside_validated_band_is_rejected():
    pick = api.candidate(
        "A @ B", "B ML", 65.0, [64.0, 63.0],
        +120, -110, bankroll=5000,
    )
    assert pick is None

def test_true_quarter_kelly_and_hard_cap():
    pct, stake, capped = api.kelly_stake(60.0, -110, 5000)
    assert 0.0 < pct <= 5.0
    assert 0.0 < stake <= 250.0
    assert stake == round(5000 * pct / 100.0, 2)

    pct_hi, stake_hi, capped_hi = api.kelly_stake(80.0, -110, 5000)
    assert pct_hi == 5.0
    assert stake_hi == 250.0
    assert capped_hi is True


def test_spread_total_probability_is_shrunk_toward_market():
    pick = api.market_candidate("A @ B", "Over 45.5", "TOTAL", 45.5, 70.0, 68.0, -110, -110, bankroll=5000)
    assert pick is not None
    assert 50.0 < pick["probability"] < 70.0
    assert pick["action"] == "LEAN"
    assert pick["stake"] == 0.0


def test_spread_total_rejects_large_model_disagreement():
    pick = api.market_candidate("A @ B", "B -3", "SPREAD", -3.0, 70.0, 58.0, -110, -110, bankroll=5000)
    assert pick is None
