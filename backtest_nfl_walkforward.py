"""Backtest walk-forward fiel al scanner de producción.

Cada semana reconstruye modelos exclusivamente con datos anteriores a esa semana.
2026 NO se evalúa aquí: queda reservado como prueba prospectiva/live.
"""

import math
import os

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from modules.nfl_calibration import empirical_residual_two_way, historico_antes, primary_with_agreement
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo


def dec(am):
    try:
        x = float(am)
        if pd.isna(x) or x == 0:
            return None
        return 1 + x / 100 if x > 0 else 1 + 100 / abs(x)
    except Exception:
        return None


def no_vig(a, b):
    da, db = dec(a), dec(b)
    if da is None or db is None:
        return None, None
    ia, ib = 1 / da, 1 / db
    s = ia + ib
    return (ia / s, ib / s) if s > 0 else (None, None)


def normalize_two_way(a, b):
    if a is None or b is None:
        return None, None
    s = float(a) + float(b)
    return (100 * float(a) / s, 100 * float(b) / s) if s > 0 else (None, None)


def select_pick(primary, supports, odd_self, odd_other):
    p = primary_with_agreement(primary, supports, max_disagreement=15.0)
    if p is None:
        return None
    mkt, _ = no_vig(odd_self, odd_other)
    d = dec(odd_self)
    if mkt is None or d is None:
        return None
    edge = (p / 100 - mkt) * 100
    ev = ((p / 100) * d - 1) * 100
    return {"p": p, "decimal": d, "edge": edge, "ev": ev} if p >= 54 and edge >= 3 and ev >= 3 else None


def settle(win, decimal):
    return decimal - 1 if win else -1.0


def logloss(y, p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -(float(y) * math.log(p) + (1 - float(y)) * math.log(1 - p))


def load_pbp():
    path = "data/historico_nfl_pbp_team_game.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def evaluate_season(raw, pbp, target_season):
    season_games = raw[raw["season"] == target_season].copy()
    weeks = sorted(pd.to_numeric(season_games["week"], errors="coerce").dropna().astype(int).unique())
    probability_rows, bet_rows = [], []

    for week in weeks:
        past = historico_antes(raw, target_season, week)
        past_pbp = historico_antes(pbp, target_season, week) if not pbp.empty else pd.DataFrame()
        wk = season_games[pd.to_numeric(season_games["week"], errors="coerce") == week].copy()

        ml = PredictorNFL_ML()
        if not ml.entrenar(past, df_pbp_team_game=past_pbp):
            continue
        elo = MotorELONFL()
        elo.actualizar_ratings(past)

        for _, r in wk.iterrows():
            if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
                continue
            home, away = r.get("home_team"), r.get("away_team")
            if not home or not away:
                continue
            if str(r.get("location", "")).strip().lower() == "neutral":
                continue

            temp = pd.to_numeric(r.get("temp"), errors="coerce")
            wind = pd.to_numeric(r.get("wind"), errors="coerce")
            roof = str(r.get("roof", "")).lower()
            dome = roof in {"dome", "closed", "indoors", "indoor"}
            hr = pd.to_numeric(r.get("home_rest"), errors="coerce")
            ar = pd.to_numeric(r.get("away_rest"), errors="coerce")
            pred = ml.predecir_contexto(
                week, home, away,
                None if pd.isna(temp) else float(temp),
                None if pd.isna(wind) else float(wind), dome,
                None if pd.isna(hr) else float(hr), None if pd.isna(ar) else float(ar),
            )
            emp = simular_nfl_montecarlo(home, away, past, r.get("total_line"), r.get("spread_line"))
            if pred is None or not emp.get("Disponible"):
                continue

            p_home, p_away = empirical_residual_two_way(
                pred.get("ML_Margen_Local_Esperado"), 0.0, ml.residuales_margen
            )
            if p_home is None or p_away is None:
                continue
            p_elo = 100 * elo.calcular_probabilidad_elo(elo.ratings.get(home, 1500), elo.ratings.get(away, 1500))
            p_emp_h, p_emp_a = normalize_two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
            if p_emp_h is None:
                continue

            hs, aws = float(r["home_score"]), float(r["away_score"])
            if hs == aws:
                continue
            y_home = int(hs > aws)
            probability_rows.append({
                "season": target_season, "week": week, "game_id": r.get("game_id"),
                "y": y_home, "p": p_home / 100.0,
                "brier": (p_home / 100.0 - y_home) ** 2,
                "logloss": logloss(y_home, p_home / 100.0),
            })

            hm, am = r.get("home_moneyline"), r.get("away_moneyline")
            if pd.isna(hm) or pd.isna(am):
                continue
            home_pick = select_pick(p_home, [p_elo, p_emp_h], hm, am)
            away_pick = select_pick(p_away, [100 - p_elo, p_emp_a], am, hm)
            choices = [("H", home_pick), ("A", away_pick)]
            choices = [x for x in choices if x[1] is not None]
            if not choices:
                continue
            side, pick = max(choices, key=lambda x: x[1]["edge"] + x[1]["ev"])
            win = (side == "H" and y_home == 1) or (side == "A" and y_home == 0)
            bet_rows.append({
                "season": target_season, "week": week, "game_id": r.get("game_id"),
                "side": side, "win": int(win), "return": settle(win, pick["decimal"]),
                "p": pick["p"], "edge": pick["edge"], "ev": pick["ev"],
            })

    probs = pd.DataFrame(probability_rows)
    bets = pd.DataFrame(bet_rows)
    result = {
        "season": target_season,
        "games_prob": len(probs),
        "brier": float(probs["brier"].mean()) if len(probs) else np.nan,
        "logloss": float(probs["logloss"].mean()) if len(probs) else np.nan,
        "picks": len(bets),
        "wins": int(bets["win"].sum()) if len(bets) else 0,
        "winrate": float(bets["win"].mean()) if len(bets) else np.nan,
        "roi": float(100 * bets["return"].mean()) if len(bets) else np.nan,
    }
    print(result)
    return result, probs, bets


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()

    all_probs, all_bets = [], []
    for season in [2023, 2024, 2025]:
        _, p, b = evaluate_season(raw, pbp, season)
        all_probs.append(p)
        all_bets.append(b)

    probs = pd.concat(all_probs, ignore_index=True) if all_probs else pd.DataFrame()
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    print("\nWALK-FORWARD TOTAL")
    print({
        "seasons": [2023, 2024, 2025], "games_prob": len(probs),
        "brier": float(probs["brier"].mean()) if len(probs) else np.nan,
        "logloss": float(probs["logloss"].mean()) if len(probs) else np.nan,
        "picks": len(bets), "wins": int(bets["win"].sum()) if len(bets) else 0,
        "winrate": float(bets["win"].mean()) if len(bets) else np.nan,
        "roi": float(100 * bets["return"].mean()) if len(bets) else np.nan,
    })
    assert len(probs) >= 300
    assert np.isfinite(probs["brier"].mean())
    assert np.isfinite(probs["logloss"].mean())


if __name__ == "__main__":
    main()
