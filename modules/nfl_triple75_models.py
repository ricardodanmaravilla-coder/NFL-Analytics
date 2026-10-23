"""Read-only predictive validation for approved Triple75 candidates.

Requires an already trained MoneylineRuntime and historical games strictly
before kickoff. Does not write Sheets or place bets.
"""
import math
from .nfl_montecarlo_sim import simular_nfl_montecarlo
from .nfl_calibration import empirical_residual_two_way


def evaluate_approved(approved, schedule, model, historical_games, *, max_gap_pp=5.0):
    indexed = {}
    for _, row in schedule.iterrows():
        indexed[(str(row["home_team"]), str(row["away_team"]))] = row
    results = []
    for item in approved:
        team, opp, venue = item["team"], item["opponent"], item["venue"]
        home, away = (team, opp) if venue == "home" else (opp, team)
        game = indexed.get((home, away))
        out = dict(item)
        out.update({"ml_probability": None, "mc_probability": None,
                    "agreement_pp": None, "expected_value_pct": None,
                    "status": "MODEL_UNAVAILABLE"})
        if game is None:
            out["reason"] = "MATCHUP_NOT_FOUND"
            results.append(out)
            continue
        date = str(game["gameday"])[:10]
        history = historical_games[
            (historical_games["gameday"].astype(str).str[:10] < date)
            & historical_games["home_score"].notna()
            & historical_games["away_score"].notna()
        ].copy()
        if history.empty:
            out["reason"] = "NO_PREGAME_HISTORY"
            results.append(out)
            continue
        week = int(game["week"])
        pred = model.predecir_contexto(week, home, away, None, None, False,
                                      game.get("home_rest"), game.get("away_rest"),
                                      season=int(game["season"]))
        mc = simular_nfl_montecarlo(home, away, history,
                     linea_ou=item["line"] if item["market"].startswith("total_") else None,
                     spread_local=(-float(item["line"]) if venue=="home" else float(item["line"])) if item["market"]=="spread" else None)
        if not pred or not mc.get("Disponible"):
            out["reason"] = "ML_OR_MC_MISSING"
            results.append(out)
            continue
        market = item["market"]
        if market == "moneyline":
            p1,p2 = empirical_residual_two_way(pred["ML_Margen_Local_Esperado"],0.0,model.residuales_margen)
            ml_p = p1 if venue=="home" else p2
            mc_p = mc["Moneyline"]["Gana Local" if venue=="home" else "Gana Visita"]
        elif market == "spread":
            threshold = -float(item["line"]) if venue=="home" else float(item["line"])
            p1,p2 = empirical_residual_two_way(pred["ML_Margen_Local_Esperado"],threshold,model.residuales_margen)
            ml_p = p1 if venue=="home" else p2
            mc_p = mc["Spread"]["Cubre Local" if venue=="home" else "Cubre Visita"]
        else:
            p1,p2 = empirical_residual_two_way(pred["ML_Puntos_Totales_Esperados"],float(item["line"]),model.residuales_total)
            over = market=="total_over"
            ml_p = p1 if over else p2
            mc_p = mc["Over_Under"]["Prob Over" if over else "Prob Under"]
        if ml_p is None or mc_p is None:
            out["reason"] = "MISSING_PROBABILITY"
            results.append(out)
            continue
        price = float(item["american_odds"])
        decimal = 1+price/100 if price>0 else 1+100/abs(price)
        edge_ev = (float(ml_p)/100*decimal-1)*100
        gap = abs(float(ml_p)-float(mc_p))
        out.update({"ml_probability":round(float(ml_p),2),
                    "mc_probability":round(float(mc_p),2),
                    "agreement_pp":round(gap,2),
                    "expected_value_pct":round(edge_ev,2),
                    "status":"MODEL_FAVORABLE" if gap<=max_gap_pp and edge_ev>0 else "TREND_ONLY",
                    "reason":"OK" if gap<=max_gap_pp and edge_ev>0 else "NO_MODEL_CONSENSUS_OR_VALUE"})
        results.append(out)
    return results
