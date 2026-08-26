"""Optimización conservadora de Moneyline sin tocar 2026.

Proceso:
- Reusa el walk-forward semanal estricto.
- Usa 2023 como desarrollo y 2024 como validación.
- 2025 se reporta sólo como confirmación final, nunca para elegir parámetros.
- Primero prueba endurecimiento de umbrales.
- Luego prueba shrinkage de la probabilidad del modelo hacia el mercado no-vig.
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp


PROB_GRID = [54.0, 55.0, 56.0, 57.0, 58.0, 60.0]
EDGE_GRID = [3.0, 4.0, 5.0, 6.0, 8.0]
EV_GRID = [3.0, 4.0, 5.0, 6.0, 8.0]
SHRINK_GRID = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]


def summarize(df):
    if df is None or df.empty:
        return {"picks": 0, "wins": 0, "winrate": np.nan, "roi": np.nan}
    return {
        "picks": int(len(df)),
        "wins": int(df["win"].sum()),
        "winrate": float(df["win"].mean()),
        "roi": float(100.0 * df["return"].mean()),
    }


def enrich_market(bets):
    x = bets.copy()
    if x.empty:
        return x
    # edge_pp = model_prob_pp - market_no_vig_prob_pp
    x["market_p"] = x["p"] - x["edge"]
    return x


def apply_rule(bets, pmin, edge_min, ev_min, shrink=1.0):
    if bets is None or bets.empty:
        return pd.DataFrame(columns=[])
    x = enrich_market(bets)
    lam = float(shrink)
    # Convex shrinkage hacia el mercado: nunca crea edge nuevo, sólo reduce sobreconfianza.
    x["p_adj"] = x["market_p"] + lam * (x["p"] - x["market_p"])
    x["edge_adj"] = x["p_adj"] - x["market_p"]
    x["ev_adj"] = ((x["p_adj"] / 100.0) * x["decimal"] - 1.0) * 100.0
    return x[
        (x["p_adj"] >= float(pmin))
        & (x["edge_adj"] >= float(edge_min))
        & (x["ev_adj"] >= float(ev_min))
    ].copy()


def robust_score(d, v, complexity_penalty=0.0):
    min_roi = min(d["roi"], v["roi"])
    mean_roi = (d["roi"] + v["roi"]) / 2.0
    min_wr = min(d["winrate"], v["winrate"])
    score = min_roi + 0.35 * mean_roi + 8.0 * (min_wr - 0.5) - complexity_penalty
    return min_roi, mean_roi, score


def choose_threshold_rule(dev, val):
    candidates = []
    for pmin in PROB_GRID:
        for edge_min in EDGE_GRID:
            for ev_min in EV_GRID:
                d = summarize(apply_rule(dev, pmin, edge_min, ev_min, shrink=1.0))
                v = summarize(apply_rule(val, pmin, edge_min, ev_min, shrink=1.0))
                if d["picks"] < 15 or v["picks"] < 12:
                    continue
                if not np.isfinite(d["roi"]) or not np.isfinite(v["roi"]):
                    continue
                complexity_penalty = 0.15 * (pmin - 54.0) + 0.08 * (edge_min - 3.0) + 0.08 * (ev_min - 3.0)
                min_roi, mean_roi, score = robust_score(d, v, complexity_penalty)
                candidates.append({
                    "pmin": pmin, "edge_min": edge_min, "ev_min": ev_min,
                    "shrink": 1.0, "dev": d, "val": v,
                    "min_roi": min_roi, "mean_roi": mean_roi, "score": score,
                })
    if not candidates:
        return None, []
    candidates.sort(key=lambda x: (x["score"], x["min_roi"], x["mean_roi"], x["dev"]["picks"] + x["val"]["picks"]), reverse=True)
    return candidates[0], candidates


def choose_shrink_rule(dev, val):
    candidates = []
    # Mantiene los umbrales base; sólo permite shrinkage conservador hacia no-vig.
    for lam in SHRINK_GRID:
        d = summarize(apply_rule(dev, 54.0, 3.0, 3.0, shrink=lam))
        v = summarize(apply_rule(val, 54.0, 3.0, 3.0, shrink=lam))
        if d["picks"] < 15 or v["picks"] < 12:
            continue
        if not np.isfinite(d["roi"]) or not np.isfinite(v["roi"]):
            continue
        # Penaliza shrinkage fuerte para exigir evidencia clara antes de alejarse de λ=1.
        complexity_penalty = 0.8 * (1.0 - lam)
        min_roi, mean_roi, score = robust_score(d, v, complexity_penalty)
        candidates.append({
            "pmin": 54.0, "edge_min": 3.0, "ev_min": 3.0,
            "shrink": lam, "dev": d, "val": v,
            "min_roi": min_roi, "mean_roi": mean_roi, "score": score,
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
    for season in [2023, 2024, 2025]:
        _, _, bets = evaluate_season(raw, pbp, season)
        season_bets[season] = bets

    dev = season_bets[2023]
    val = season_bets[2024]
    test = season_bets[2025]

    threshold_choice, threshold_candidates = choose_threshold_rule(dev, val)
    shrink_choice, shrink_candidates = choose_shrink_rule(dev, val)
    if threshold_choice is None or shrink_choice is None:
        raise SystemExit("No robust Moneyline candidate with sufficient volume")

    print("\nTHRESHOLD RULE SELECTED WITHOUT 2025")
    print({
        "pmin": threshold_choice["pmin"], "edge_min": threshold_choice["edge_min"],
        "ev_min": threshold_choice["ev_min"], "2023_dev": threshold_choice["dev"],
        "2024_validation": threshold_choice["val"],
        "2025_untouched_test": summarize(apply_rule(test, threshold_choice["pmin"], threshold_choice["edge_min"], threshold_choice["ev_min"], 1.0)),
    })

    print("\nMARKET SHRINKAGE SELECTED WITHOUT 2025")
    lam = shrink_choice["shrink"]
    shrink_test = summarize(apply_rule(test, 54.0, 3.0, 3.0, shrink=lam))
    print({
        "lambda_model": lam, "lambda_market": round(1.0 - lam, 2),
        "2023_dev": shrink_choice["dev"], "2024_validation": shrink_choice["val"],
        "2025_untouched_test": shrink_test,
    })

    base = pd.concat([season_bets[2023], season_bets[2024], season_bets[2025]], ignore_index=True)
    print("BASE_54_3_3", summarize(apply_rule(base, 54.0, 3.0, 3.0, 1.0)))
    print("SHRINK_SELECTED", summarize(apply_rule(base, 54.0, 3.0, 3.0, lam)))

    print("\nSHRINKAGE GRID DEV/VALIDATION")
    for row in shrink_candidates:
        print({
            "lambda_model": row["shrink"],
            "dev_roi": round(row["dev"]["roi"], 3), "dev_picks": row["dev"]["picks"],
            "val_roi": round(row["val"]["roi"], 3), "val_picks": row["val"]["picks"],
            "min_roi": round(row["min_roi"], 3), "score": round(row["score"], 3),
        })

    assert threshold_choice["dev"]["picks"] >= 15
    assert threshold_choice["val"]["picks"] >= 12
    assert shrink_choice["dev"]["picks"] >= 15
    assert shrink_choice["val"]["picks"] >= 12
    assert 2026 not in set(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int).unique())


if __name__ == "__main__":
    main()
