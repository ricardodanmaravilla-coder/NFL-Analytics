import pandas as pd
import nfl_data_py as nfl

import backtest_nfl_pbp_v1 as bt


GROUPS = {
    "EARLY": ["early_down_epa", "early_down_success", "early_down_epa_allowed"],
    "NEUTRAL": ["neutral_pass_rate"],
    "REDZONE": ["redzone_epa", "redzone_success", "redzone_epa_allowed"],
    "LATE": ["third_fourth_epa", "late_down_success", "late_down_epa_allowed"],
    "EARLY_NEUTRAL": ["early_down_epa", "early_down_success", "early_down_epa_allowed", "neutral_pass_rate"],
    "EARLY_REDZONE": ["early_down_epa", "early_down_success", "early_down_epa_allowed", "redzone_epa", "redzone_success", "redzone_epa_allowed"],
    "REDZONE_LATE": ["redzone_epa", "redzone_success", "redzone_epa_allowed", "third_fourth_epa", "late_down_success", "late_down_epa_allowed"],
}


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = pd.read_csv("data/historico_nfl_pbp_team_game.csv")

    original = list(bt.SITUATIONAL_METRICS)
    hybrid = bt.evaluate(raw, pbp, "HYBRID")
    rows = []
    try:
        for name, metrics in GROUPS.items():
            bt.SITUATIONAL_METRICS = list(metrics)
            result = bt.evaluate(raw, pbp, "SITUATIONAL")
            row = {
                "group": name,
                "features": ",".join(metrics),
                "test": result["test"],
                "margin_mae": result["margin_mae"],
                "winner_acc": result["winner_acc"],
                "ats_acc": result["ats_acc"],
                "ml_roi": result["ml_roi"],
                "delta_mae": result["margin_mae"] - hybrid["margin_mae"],
                "delta_winner_pp": 100 * (result["winner_acc"] - hybrid["winner_acc"]),
                "delta_ats_pp": 100 * (result["ats_acc"] - hybrid["ats_acc"]),
                "delta_ml_roi_pp": result["ml_roi"] - hybrid["ml_roi"],
            }
            rows.append(row)
            print("ABLATION", row)
    finally:
        bt.SITUATIONAL_METRICS = original

    out = pd.DataFrame(rows).sort_values(["delta_mae", "delta_winner_pp"], ascending=[True, False])
    print("\nSITUATIONAL ABLATION RANKING")
    print(out.to_string(index=False))

    # Conservative acceptance: require a real MAE improvement and no material
    # winner/ROI regression. This prevents selecting a feature group on one noisy metric.
    viable = out[
        (out["delta_mae"] <= -0.03)
        & (out["delta_winner_pp"] >= -0.75)
        & (out["delta_ml_roi_pp"] >= -3.0)
    ].copy()
    if viable.empty:
        print("SELECTED=NONE — HYBRID remains production champion")
    else:
        best = viable.iloc[0]
        print(f"SELECTED={best['group']} delta_mae={best['delta_mae']:.4f} delta_winner_pp={best['delta_winner_pp']:.2f} delta_ml_roi_pp={best['delta_ml_roi_pp']:.2f}")

    assert hybrid["test"] >= 150
    assert out["test"].min() == hybrid["test"]


if __name__ == "__main__":
    main()
