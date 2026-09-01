"""Auditoria neutral para detectar sesgo de seleccion en NFL Analytics.

No modifica produccion ni activa mercados. Compara Moneyline con la disponibilidad
y calidad de Spread/Total, y segmenta Moneyline por favorito/underdog y local/visita.
Usa exclusivamente el walk-forward temporal 2023-2025 ya existente.
"""
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from backtest_nfl_walkforward import evaluate_season, load_pbp


def summarize(df):
    if df is None or df.empty:
        return {"n": 0, "wins": 0, "winrate": np.nan, "roi": np.nan, "avg_edge": np.nan, "avg_ev": np.nan}
    return {
        "n": int(len(df)),
        "wins": int(df["win"].sum()),
        "winrate": float(df["win"].mean()),
        "roi": float(100.0 * df["return"].mean()),
        "avg_edge": float(df["edge"].mean()),
        "avg_ev": float(df["ev"].mean()),
    }


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = load_pbp()

    bets = []
    for season in (2023, 2024, 2025):
        _, _, b = evaluate_season(raw, pbp, season)
        if not b.empty:
            bets.append(b)
    bets = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
    if bets.empty:
        raise RuntimeError("No se generaron picks Moneyline en walk-forward")

    # Une solo informacion de mercado conocida para clasificar, no para seleccionar.
    market_cols = [c for c in ["season", "week", "game_id", "home_moneyline", "away_moneyline", "spread_line", "total_line"] if c in raw.columns]
    market = raw[market_cols].drop_duplicates(subset=["season", "week", "game_id"])
    x = bets.merge(market, on=["season", "week", "game_id"], how="left")
    chosen_odds = np.where(x["side"].eq("H"), x["home_moneyline"], x["away_moneyline"])
    x["favorite"] = pd.to_numeric(chosen_odds, errors="coerce") < 0
    x["home_side"] = x["side"].eq("H")

    print("\n=== MONEYLINE SEGMENT AUDIT ===")
    for season in (2023, 2024, 2025):
        s = x[x["season"] == season]
        print("SEASON", season, "ALL", summarize(s))
        print("SEASON", season, "FAVORITE", summarize(s[s["favorite"]]))
        print("SEASON", season, "UNDERDOG", summarize(s[~s["favorite"]]))
        print("SEASON", season, "HOME", summarize(s[s["home_side"]]))
        print("SEASON", season, "AWAY", summarize(s[~s["home_side"]]))

    print("TOTAL ALL", summarize(x))
    print("TOTAL FAVORITE", summarize(x[x["favorite"]]))
    print("TOTAL UNDERDOG", summarize(x[~x["favorite"]]))
    print("TOTAL HOME", summarize(x[x["home_side"]]))
    print("TOTAL AWAY", summarize(x[~x["home_side"]]))

    # Diagnostico de por que no es valido concluir aun que ML supera Spread/Total:
    # el backtest de produccion solo genera apuestas Moneyline. Medimos cobertura de
    # lineas para decidir si una auditoria simetrica de esos mercados es factible.
    eval_raw = raw[raw["season"].isin([2023, 2024, 2025])]
    spread_cov = float(pd.to_numeric(eval_raw.get("spread_line"), errors="coerce").notna().mean()) if "spread_line" in eval_raw else 0.0
    total_cov = float(pd.to_numeric(eval_raw.get("total_line"), errors="coerce").notna().mean()) if "total_line" in eval_raw else 0.0
    ml_cov = float((pd.to_numeric(eval_raw.get("home_moneyline"), errors="coerce").notna() & pd.to_numeric(eval_raw.get("away_moneyline"), errors="coerce").notna()).mean()) if {"home_moneyline", "away_moneyline"}.issubset(eval_raw.columns) else 0.0
    print("\n=== MARKET DATA COVERAGE ===")
    print({"moneyline_two_way": ml_cov, "spread_line": spread_cov, "total_line": total_cov})
    print("AUDIT_CONCLUSION: produccion actual esta estructuralmente gated a Moneyline; no interpretar ausencia de Spread/Total como superioridad estadistica hasta construir backtests simetricos con precios/settlement equivalentes.")


if __name__ == "__main__":
    main()
