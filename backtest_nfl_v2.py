import math

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_xgb_engine import PredictorXGBoostSpread


def main():
    seasons = [2021, 2022, 2023, 2024, 2025]
    raw = nfl.import_schedules(seasons)
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()

    builder = PredictorNFL_ML()
    feats = builder.construir_features_pregame(raw)
    market_cols = [c for c in ["game_id", "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"] if c in raw.columns]
    df = feats.merge(raw[market_cols], on="game_id", how="left")

    feature_cols = [
        "week", "home_altitude", "temp", "wind", "is_dome", "temp_missing", "wind_missing",
        "home_rest", "away_rest",
        "home_off_5", "home_def_5", "home_margin_5", "home_total_5",
        "home_off_17", "home_def_17", "home_margin_17", "home_total_17", "home_score_sd_17",
        "away_off_5", "away_def_5", "away_margin_5", "away_total_5",
        "away_off_17", "away_def_17", "away_margin_17", "away_total_17", "away_score_sd_17",
    ]
    df = df.dropna(subset=feature_cols + ["puntos_totales", "margen_local"]).reset_index(drop=True)
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut], df.iloc[cut:]

    total_model = RandomForestRegressor(n_estimators=250, max_depth=9, min_samples_leaf=6, random_state=42, n_jobs=1)
    margin_model = RandomForestRegressor(n_estimators=250, max_depth=9, min_samples_leaf=6, random_state=43, n_jobs=1)
    total_model.fit(train[feature_cols], train["puntos_totales"])
    margin_model.fit(train[feature_cols], train["margen_local"])

    pred_total = total_model.predict(test[feature_cols])
    pred_margin = margin_model.predict(test[feature_cols])
    mae_total = mean_absolute_error(test["puntos_totales"], pred_total)
    rmse_total = math.sqrt(mean_squared_error(test["puntos_totales"], pred_total))
    mae_margin = mean_absolute_error(test["margen_local"], pred_margin)
    rmse_margin = math.sqrt(mean_squared_error(test["margen_local"], pred_margin))

    winner_mask = test["margen_local"] != 0
    winner_acc = float(np.mean((pred_margin[winner_mask] > 0) == (test.loc[winner_mask, "margen_local"].to_numpy() > 0)))
    home_base = float(np.mean(test.loc[winner_mask, "margen_local"] > 0))

    print(f"games={len(df)} train={len(train)} test={len(test)}")
    print(f"total_mae={mae_total:.4f} total_rmse={rmse_total:.4f}")
    print(f"margin_mae={mae_margin:.4f} margin_rmse={rmse_margin:.4f}")
    print(f"winner_accuracy={winner_acc:.4f} home_win_baseline={home_base:.4f}")

    if "spread_line" in test.columns:
        m = test["spread_line"].notna()
        actual_adj = test.loc[m, "margen_local"].to_numpy() - test.loc[m, "spread_line"].to_numpy()
        nonpush = actual_adj != 0
        if np.any(nonpush):
            model_home_cover = pred_margin[m][nonpush] > test.loc[m, "spread_line"].to_numpy()[nonpush]
            actual_home_cover = actual_adj[nonpush] > 0
            ats_acc = float(np.mean(model_home_cover == actual_home_cover))
            print(f"ats_games={len(actual_home_cover)} ats_accuracy={ats_acc:.4f}")

    if "total_line" in test.columns:
        m = test["total_line"].notna()
        actual_delta = test.loc[m, "puntos_totales"].to_numpy() - test.loc[m, "total_line"].to_numpy()
        nonpush = actual_delta != 0
        if np.any(nonpush):
            model_over = pred_total[m][nonpush] > test.loc[m, "total_line"].to_numpy()[nonpush]
            actual_over = actual_delta[nonpush] > 0
            ou_acc = float(np.mean(model_over == actual_over))
            print(f"ou_games={len(actual_over)} ou_accuracy={ou_acc:.4f}")

    xgb = PredictorXGBoostSpread()
    xgb_ok = xgb.entrenar(raw)
    print(f"xgb_trained={xgb_ok} xgb_oos_accuracy={xgb.oos_accuracy} xgb_oos_brier={xgb.oos_brier}")

    # Fail only on structural problems, not on predictive performance; performance
    # remains visible for human review before merge.
    assert len(test) >= 150
    assert np.isfinite(mae_total) and np.isfinite(mae_margin)
    if xgb_ok:
        assert xgb.oos_accuracy is not None and xgb.oos_brier is not None


if __name__ == "__main__":
    main()
