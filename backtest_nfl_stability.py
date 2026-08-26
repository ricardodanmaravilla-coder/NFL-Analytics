"""Optimización conservadora de umbrales Moneyline sin tocar 2026.

Proceso:
- Reusa el walk-forward semanal estricto.
- Usa 2023 como desarrollo y 2024 como validación.
- 2025 se reporta sólo como confirmación final, nunca para elegir umbrales.
- Sólo permite endurecer los mínimos actuales (54% / 3pp / 3% EV).
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp


PROB_GRID = [54.0, 55.0, 56.0, 57.0, 58.0, 60.0]
EDGE_GRID = [3.0, 4.0, 5.0, 6.0, 8.0]
EV_GRID = [3.0, 4.0, 5.0, 6.0, 8.0]


def summarize(df):
    if df is None or df.empty:
        return {"picks": 0, "wins": 0, "winrate": np.nan, "roi": np.nan}
    return {
        "picks": int(len(df)),
        "wins": int(df["win"].sum()),
        "winrate": float(df["win"].mean()),
        "roi": float(100.0 * df["return"].mean()),
    }


def apply_rule(bets, pmin, edge_min, ev_min):
    return bets[
        (bets["p"] >= float(pmin))
        & (bets["edge"] >= float(edge_min))
        & (bets["ev"] >= float(ev_min))
    ].copy()


def choose_rule(dev, val):
    candidates = []
    for pmin in PROB_GRID:
        for edge_min in EDGE_GRID:
            for ev_min in EV_GRID:
                d = summarize(apply_rule(dev, pmin, edge_min, ev_min))
                v = summarize(apply_rule(val, pmin, edge_min, ev_min))
                # Evita reglas de muestra ridículamente pequeña.
                if d["picks"] < 15 or v["picks"] < 12:
                    continue
                if not np.isfinite(d["roi"]) or not np.isfinite(v["roi"]):
                    continue
                # Selección por robustez, no por el pico de un solo año.
                min_roi = min(d["roi"], v["roi"])
                mean_roi = (d["roi"] + v["roi"]) / 2.0
                min_wr = min(d["winrate"], v["winrate"])
                # Penaliza endurecimiento excesivo y favorece volumen reproducible.
                complexity_penalty = 0.15 * (pmin - 54.0) + 0.08 * (edge_min - 3.0) + 0.08 * (ev_min - 3.0)
                score = min_roi + 0.35 * mean_roi + 8.0 * (min_wr - 0.5) - complexity_penalty
                candidates.append({
                    "pmin": pmin,
                    "edge_min": edge_min,
                    "ev_min": ev_min,
                    "dev": d,
                    "val": v,
                    "min_roi": min_roi,
                    "mean_roi": mean_roi,
                    "score": score,
                })
    if not candidates:
        return None, []
    candidates.sort(key=lambda x: (x["score"], x["min_roi"], x["mean_roi"], x["dev"]["picks"] + x["val"]["picks"]), reverse=True)
    return candidates[0], candidates


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()

    season_bets = {}
    season_probs = {}
    for season in [2023, 2024, 2025]:
        _, probs, bets = evaluate_season(raw, pbp, season)
        season_bets[season] = bets
        season_probs[season] = probs

    dev = season_bets[2023]
    val = season_bets[2024]
    test = season_bets[2025]
    chosen, candidates = choose_rule(dev, val)
    if chosen is None:
        raise SystemExit("No robust threshold candidate with sufficient volume")

    pmin = chosen["pmin"]
    edge_min = chosen["edge_min"]
    ev_min = chosen["ev_min"]
    test_summary = summarize(apply_rule(test, pmin, edge_min, ev_min))

    print("\nSTABILITY RULE SELECTED WITHOUT 2025")
    print({
        "pmin": pmin,
        "edge_min": edge_min,
        "ev_min": ev_min,
        "2023_dev": chosen["dev"],
        "2024_validation": chosen["val"],
        "2025_untouched_test": test_summary,
    })

    base = pd.concat([season_bets[2023], season_bets[2024], season_bets[2025]], ignore_index=True)
    robust = apply_rule(base, pmin, edge_min, ev_min)
    print("BASE_54_3_3", summarize(base))
    print("ROBUST_SELECTED", summarize(robust))

    print("\nTOP 10 DEV/VALIDATION RULES")
    for row in candidates[:10]:
        print({
            "pmin": row["pmin"], "edge_min": row["edge_min"], "ev_min": row["ev_min"],
            "dev_roi": round(row["dev"]["roi"], 3), "dev_picks": row["dev"]["picks"],
            "val_roi": round(row["val"]["roi"], 3), "val_picks": row["val"]["picks"],
            "min_roi": round(row["min_roi"], 3), "score": round(row["score"], 3),
        })

    # Guardrails metodológicos, no una exigencia artificial de ROI positivo en 2025.
    assert chosen["dev"]["picks"] >= 15
    assert chosen["val"]["picks"] >= 12
    assert 2026 not in set(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int).unique())


if __name__ == "__main__":
    main()
