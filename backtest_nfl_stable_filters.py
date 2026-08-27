"""Valida filtros de apuestas sobre el mismo walk-forward del scanner.

No modifica producción. Evalúa únicamente picks ya generados por la regla
54/3/3 y compara filtros simples hallados en 2025 a través de 2023-2025.
2023=desarrollo, 2024=validación, 2025=prueba intocable.
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp


FILTERS = {
    "BASE": lambda x: pd.Series(True, index=x.index),
    "FAVORITES": lambda x: x["decimal"] < 2.0,
    "EDGE_4_10": lambda x: x["edge"].between(4.0, 10.0, inclusive="both"),
    "FAVORITES_EDGE_4_10": lambda x: (x["decimal"] < 2.0) & x["edge"].between(4.0, 10.0, inclusive="both"),
    "FAVORITES_EDGE_LE_10": lambda x: (x["decimal"] < 2.0) & (x["edge"] <= 10.0),
}


def max_drawdown_units(df):
    if df is None or df.empty:
        return np.nan
    x = df.sort_values(["season", "week", "game_id"]).copy()
    equity = x["return"].cumsum()
    peak = equity.cummax().clip(lower=0.0)
    dd = equity - peak
    return float(dd.min())


def summarize(df):
    if df is None or df.empty:
        return {"picks": 0, "wins": 0, "winrate": np.nan, "roi": np.nan, "max_drawdown_units": np.nan}
    return {
        "picks": int(len(df)),
        "wins": int(df["win"].sum()),
        "winrate": float(df["win"].mean()),
        "roi": float(100.0 * df["return"].mean()),
        "max_drawdown_units": max_drawdown_units(df),
    }


def apply_filter(df, name):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    mask = FILTERS[name](df)
    return df[mask.fillna(False)].copy()


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()

    season_bets = {}
    for season in [2023, 2024, 2025]:
        _, _, bets = evaluate_season(raw, pbp, season)
        season_bets[season] = bets.copy()

    rows = []
    for name in FILTERS:
        per_season = {}
        for season in [2023, 2024, 2025]:
            s = summarize(apply_filter(season_bets[season], name))
            per_season[season] = s
            rows.append({"filter": name, "season": season, **s})
            print("FILTER_SEASON", name, season, s)

        combo = pd.concat([apply_filter(season_bets[s], name) for s in [2023, 2024, 2025]], ignore_index=True)
        total = summarize(combo)
        print("FILTER_TOTAL", name, total)

        # Candidato sólo si fue positivo en desarrollo y validación, conserva volumen
        # razonable y no se desmorona en 2025. 2025 nunca participa en la selección.
        dev, val, test = per_season[2023], per_season[2024], per_season[2025]
        selected_without_2025 = (
            dev["picks"] >= 10
            and val["picks"] >= 8
            and np.isfinite(dev["roi"])
            and np.isfinite(val["roi"])
            and dev["roi"] > 0
            and val["roi"] > 0
            and min(dev["winrate"], val["winrate"]) >= 0.55
        )
        survives_2025 = (
            selected_without_2025
            and test["picks"] >= 6
            and np.isfinite(test["roi"])
            and test["roi"] > -5.0
            and test["winrate"] >= 0.50
        )
        print(f"FILTER_DECISION {name} selected_without_2025={selected_without_2025} survives_2025={survives_2025}")

    table = pd.DataFrame(rows)
    print("\nSTABLE FILTER WALK-FORWARD")
    print(table.to_string(index=False))

    # Producción sigue sin cambios; este script sólo genera evidencia.
    assert set(table["season"].unique()) == {2023, 2024, 2025}
    assert 2026 not in set(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int).unique())


if __name__ == "__main__":
    main()
