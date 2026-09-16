"""Audit walk-forward de umbrales de probabilidad para NFL.

Reutiliza exactamente el backtest de producción (54% base, edge>=3pp, EV>=3%)
y re-filtra sus picks por probabilidad mínima para no reentrenar ni cambiar otra
regla. Reporta todos los candidatos y el subconjunto de favoritos que producción
convierte en BET.
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp

THRESHOLDS = [54, 56, 58, 60, 62, 65]
SEASONS = [2023, 2024, 2025]


def max_drawdown_unit_stakes(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    equity = 1.0 + bets["return"].astype(float).cumsum()
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks.replace(0, np.nan)
    return float(100 * dd.min()) if len(dd) else float("nan")


def summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"picks": 0, "wins": 0, "winrate": np.nan, "roi": np.nan, "profit_u": 0.0, "max_dd_pct": np.nan}
    ordered = bets.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return {
        "picks": int(len(ordered)),
        "wins": int(ordered["win"].sum()),
        "winrate": float(100 * ordered["win"].mean()),
        "roi": float(100 * ordered["return"].mean()),
        "profit_u": float(ordered["return"].sum()),
        "max_dd_pct": max_drawdown_unit_stakes(ordered),
    }


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()

    season_bets = {}
    for season in SEASONS:
        _, _, bets = evaluate_season(raw, pbp, season)
        season_bets[season] = bets.copy()

    rows = []
    for threshold in THRESHOLDS:
        for season in SEASONS + ["TOTAL"]:
            base = pd.concat([season_bets[s] for s in SEASONS], ignore_index=True) if season == "TOTAL" else season_bets[season]
            filt = base[pd.to_numeric(base["p"], errors="coerce") >= threshold].copy()
            scopes = [
                ("ALL_CANDIDATES", filt),
                ("PRODUCTION_FAVORITES", filt[pd.to_numeric(filt["decimal"], errors="coerce") < 2.0].copy()),
            ]
            for scope, sample in scopes:
                rows.append({"threshold": threshold, "season": season, "scope": scope, **summarize(sample)})

    out = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print("\nNFL PROBABILITY THRESHOLD AUDIT")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    holdout = out[(out["season"] == 2025) & (out["scope"] == "PRODUCTION_FAVORITES")].copy()
    total = out[(out["season"] == "TOTAL") & (out["scope"] == "PRODUCTION_FAVORITES")].copy()
    print("\n2025 HOLDOUT - PRODUCTION FAVORITES")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n2023-2025 TOTAL - PRODUCTION FAVORITES")
    print(total.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    assert set(holdout["threshold"]) == set(THRESHOLDS)
    assert int(out[(out["threshold"] == 54) & (out["season"] == "TOTAL") & (out["scope"] == "ALL_CANDIDATES")]["picks"].iloc[0]) >= 80


if __name__ == "__main__":
    main()
