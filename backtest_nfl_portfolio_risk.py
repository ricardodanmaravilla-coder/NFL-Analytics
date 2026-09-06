"""Evalúa riesgo de cartera Moneyline sin tocar producción ni usar 2026.

2023=desarrollo, 2024=validación, 2025=prueba intocable. Parte de los picks
walk-forward ya filtrados por 54/3/3 y conserva sólo favoritos, igual que BET automático.
"""
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp

MAX_BETS_GRID = [1, 2, 3, 5, 99]
EXPOSURE_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 1.0]


def quarter_kelly(p_pct, decimal, cap=0.05):
    p = float(p_pct) / 100.0
    b = float(decimal) - 1.0
    if b <= 0:
        return 0.0
    full = max(0.0, (b * p - (1.0 - p)) / b)
    return min(0.25 * full, cap)


def portfolio(df, max_bets, max_exposure, initial=1.0):
    if df is None or df.empty:
        return {"bets": 0, "weeks": 0, "profit": 0.0, "roi_staked": np.nan,
                "ending_bankroll": initial, "max_drawdown_pct": 0.0, "max_week_exposure": 0.0}
    x = df[df["decimal"] < 2.0].copy()  # producción BET sólo favoritos
    if x.empty:
        return {"bets": 0, "weeks": 0, "profit": 0.0, "roi_staked": np.nan,
                "ending_bankroll": initial, "max_drawdown_pct": 0.0, "max_week_exposure": 0.0}
    x["rank_score"] = 1.5 * x["edge"] + x["ev"]
    bankroll = float(initial)
    peak = bankroll
    max_dd = 0.0
    total_staked = 0.0
    bet_count = 0
    week_count = 0
    max_week_seen = 0.0
    for (_, _), wk in x.groupby(["season", "week"], sort=True):
        wk = wk.sort_values(["rank_score", "edge", "ev"], ascending=False).head(int(max_bets))
        remaining = float(max_exposure)
        week_fraction = 0.0
        positions = []
        for _, r in wk.iterrows():
            f = quarter_kelly(r["p"], r["decimal"])
            f = min(f, remaining)
            if f <= 1e-12:
                continue
            positions.append((f, int(r["win"]), float(r["decimal"])))
            remaining -= f
            week_fraction += f
            if remaining <= 1e-12:
                break
        if not positions:
            continue
        start_bank = bankroll
        pnl = 0.0
        for f, win, dec in positions:
            stake = start_bank * f
            total_staked += stake
            pnl += stake * ((dec - 1.0) if win else -1.0)
            bet_count += 1
        bankroll += pnl
        week_count += 1
        max_week_seen = max(max_week_seen, week_fraction)
        peak = max(peak, bankroll)
        if peak > 0:
            max_dd = min(max_dd, (bankroll - peak) / peak)
    profit = bankroll - initial
    return {
        "bets": bet_count,
        "weeks": week_count,
        "profit": profit,
        "roi_staked": (100.0 * profit / total_staked) if total_staked > 0 else np.nan,
        "ending_bankroll": bankroll,
        "max_drawdown_pct": 100.0 * max_dd,
        "max_week_exposure": 100.0 * max_week_seen,
    }


def candidate_score(dev, val):
    if dev["bets"] < 12 or val["bets"] < 10:
        return -np.inf
    if dev["profit"] <= 0 or val["profit"] <= 0:
        return -np.inf
    # Prima robustez y castiga drawdown; no usa 2025 para seleccionar.
    growth = min(dev["profit"], val["profit"])
    drawdown_penalty = 0.02 * (abs(dev["max_drawdown_pct"]) + abs(val["max_drawdown_pct"]))
    return growth - drawdown_penalty


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()
    bets = {}
    for season in [2023, 2024, 2025]:
        _, _, b = evaluate_season(raw, pbp, season)
        bets[season] = b

    rows = []
    for max_bets in MAX_BETS_GRID:
        for exposure in EXPOSURE_GRID:
            dev = portfolio(bets[2023], max_bets, exposure)
            val = portfolio(bets[2024], max_bets, exposure)
            score = candidate_score(dev, val)
            rows.append({"max_bets": max_bets, "max_exposure": exposure, "score": score,
                         "dev": dev, "val": val})
            print("RISK_RULE", max_bets, exposure, "DEV", dev, "VAL", val, "score", score)
    valid = [r for r in rows if np.isfinite(r["score"])]
    if not valid:
        raise SystemExit("No robust portfolio rule found")
    valid.sort(key=lambda r: (r["score"], -r["max_exposure"], -r["max_bets"]), reverse=True)
    best = valid[0]
    test = portfolio(bets[2025], best["max_bets"], best["max_exposure"])
    baseline = portfolio(pd.concat([bets[2023], bets[2024], bets[2025]], ignore_index=True), 99, 1.0)
    selected_all = portfolio(pd.concat([bets[2023], bets[2024], bets[2025]], ignore_index=True), best["max_bets"], best["max_exposure"])
    print("SELECTED_WITHOUT_2025", best)
    print("2025_UNTOUCHED", test)
    print("BASELINE_ALL", baseline)
    print("SELECTED_ALL", selected_all)
    assert 2026 not in set(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int).unique())
    assert best["max_exposure"] <= 1.0


if __name__ == "__main__":
    main()
