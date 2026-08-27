import math

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import backtest_nfl_pbp_v1 as bt
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_pbp_engine import construir_pbp_pregame


CHALLENGER_METRICS = [
    "early_down_epa",
    "early_down_success",
    "early_down_epa_allowed",
    "neutral_pass_rate",
]


def p_gt(mean, threshold, sigma):
    if sigma is None or sigma <= 0:
        return None
    z = (float(mean) - float(threshold)) / float(sigma)
    return 50 * (1 + math.erf(z / math.sqrt(2)))


def max_drawdown(returns):
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        equity += float(r)
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
    return 100 * mdd


def build_predictions(raw, pbp):
    builder = bt.PredictorNFL_ML()
    feat = builder.construir_features_pregame(raw, pbp)
    situ = construir_pbp_pregame(raw, pbp, metrics=CHALLENGER_METRICS)
    feat = feat.merge(situ, on="game_id", how="left")

    market_cols = [
        "game_id", "season", "week", "home_team", "away_team", "spread_line",
        "home_moneyline", "away_moneyline", "home_score", "away_score",
    ]
    df = feat.merge(raw[[c for c in market_cols if c in raw.columns]], on="game_id", how="left")
    df = df.dropna(subset=["home_off_epa_play_4", "away_off_epa_play_4"]).copy()

    hybrid_features = bt.BASE_FEATURES + [c for c in bt.pbp_subset("FULL") if c in df.columns]
    challenger_extra = [
        f"{side}_{metric}_{w}"
        for side in ["home", "away"]
        for metric in CHALLENGER_METRICS
        for w in [4, 8]
        if f"{side}_{metric}_{w}" in df.columns
    ]
    challenger_features = hybrid_features + challenger_extra
    required = list(dict.fromkeys(hybrid_features + challenger_features + ["margen_local"]))
    df = df.dropna(subset=required).copy()

    train = df[df["season"] <= 2024].copy()
    test = df[df["season"] == 2025].sort_values(["week", "game_id"]).copy()
    assert len(train) > 600 and len(test) >= 150

    cal = int(len(train) * 0.8)
    hybrid = RandomForestRegressor(
        n_estimators=250, max_depth=9, min_samples_leaf=6, random_state=43, n_jobs=1
    )
    challenger = RandomForestRegressor(
        n_estimators=250, max_depth=9, min_samples_leaf=6, random_state=43, n_jobs=1
    )

    hybrid.fit(train[hybrid_features].iloc[:cal], train["margen_local"].iloc[:cal])
    challenger.fit(train[challenger_features].iloc[:cal], train["margen_local"].iloc[:cal])
    hybrid_res = train["margen_local"].iloc[cal:].to_numpy() - hybrid.predict(train[hybrid_features].iloc[cal:])
    challenger_res = train["margen_local"].iloc[cal:].to_numpy() - challenger.predict(train[challenger_features].iloc[cal:])
    sigma_h = float(np.std(hybrid_res, ddof=1))
    sigma_c = float(np.std(challenger_res, ddof=1))

    hybrid.fit(train[hybrid_features], train["margen_local"])
    challenger.fit(train[challenger_features], train["margen_local"])
    test["hybrid_margin"] = hybrid.predict(test[hybrid_features])
    test["challenger_margin"] = challenger.predict(test[challenger_features])
    test["hybrid_prob_home"] = [p_gt(x, 0, sigma_h) for x in test["hybrid_margin"]]
    test["challenger_prob_home"] = [p_gt(x, 0, sigma_c) for x in test["challenger_margin"]]
    test["agree_winner"] = (test["hybrid_margin"] > 0) == (test["challenger_margin"] > 0)
    test["agreement_strength"] = np.minimum(abs(test["hybrid_margin"]), abs(test["challenger_margin"]))
    return test, sigma_h


def generate_hybrid_picks(raw, test, sigma_h):
    picks = []
    for week, wk in test.groupby("week", sort=True):
        past = raw[(raw["season"] < 2025) | ((raw["season"] == 2025) & (raw["week"] < week))].copy()
        elo = MotorELONFL()
        elo.actualizar_ratings(past)
        for _, r in wk.iterrows():
            hm, am = r.get("home_moneyline"), r.get("away_moneyline")
            if pd.isna(hm) or pd.isna(am):
                continue
            home, away = r["home_team"], r["away_team"]
            emp = simular_nfl_montecarlo(home, away, past, None, r.get("spread_line"))
            if not emp.get("Disponible"):
                continue
            elo_h = 100 * elo.calcular_probabilidad_elo(elo.ratings.get(home, 1500), elo.ratings.get(away, 1500))
            ml_h = p_gt(r["hybrid_margin"], 0, sigma_h)
            emp_h, emp_a = bt.norm2(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
            ch = bt.choose([elo_h, ml_h, emp_h], hm, am, 54)
            ca = bt.choose([100 - elo_h, 100 - ml_h, emp_a], am, hm, 54)
            choices = [x for x in [("H", ch, hm), ("A", ca, am)] if x[1] is not None]
            if not choices:
                continue
            side, c, odds = max(choices, key=lambda x: x[1][2] + x[1][3])
            actual_home_win = r["home_score"] > r["away_score"]
            win = (side == "H" and actual_home_win) or (side == "A" and not actual_home_win)
            push = r["home_score"] == r["away_score"]
            ret = bt.settle(win, push, c[1])
            pick_team = home if side == "H" else away
            price = float(odds)
            favorite = price < 0
            challenger_side = "H" if r["challenger_margin"] > 0 else "A"
            hybrid_side = "H" if r["hybrid_margin"] > 0 else "A"
            picks.append({
                "season": 2025,
                "week": int(r["week"]),
                "game_id": r["game_id"],
                "pick_side": side,
                "pick_team": pick_team,
                "price": price,
                "favorite": favorite,
                "home_pick": side == "H",
                "win": bool(win),
                "return": ret,
                "edge": float(c[2]),
                "ev": float(c[3]),
                "prob": float(c[0]),
                "hybrid_margin": float(r["hybrid_margin"]),
                "challenger_margin": float(r["challenger_margin"]),
                "model_agree": hybrid_side == challenger_side,
                "pick_agree": side == challenger_side,
                "agreement_strength": float(r["agreement_strength"]),
            })
    return pd.DataFrame(picks)


def summarize(name, df):
    if df.empty:
        out = {"name": name, "picks": 0, "wins": 0, "winrate": np.nan, "roi": np.nan, "max_drawdown_pct": np.nan}
    else:
        out = {
            "name": name,
            "picks": len(df),
            "wins": int(df["win"].sum()),
            "winrate": float(df["win"].mean()),
            "roi": 100 * float(df["return"].sum()) / len(df),
            "max_drawdown_pct": max_drawdown(df.sort_values(["week", "game_id"])["return"]),
        }
    print("SUMMARY", out)
    return out


def segment(picks, col):
    rows = []
    for value, g in picks.groupby(col, dropna=False):
        if len(g) < 4:
            continue
        s = summarize(f"{col}={value}", g)
        rows.append({"segment": col, "value": value, **s})
    return rows


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw["result"].notna()].copy()
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    pbp = pd.read_csv("data/historico_nfl_pbp_team_game.csv")

    test, sigma_h = build_predictions(raw, pbp)
    picks = generate_hybrid_picks(raw, test, sigma_h)
    assert len(picks) >= 20

    base = summarize("HYBRID_ALL", picks)
    agree = summarize("HYBRID_PICK_AGREES_WITH_CHALLENGER", picks[picks["pick_agree"]])
    disagree = summarize("HYBRID_PICK_DISAGREES_WITH_CHALLENGER", picks[~picks["pick_agree"]])

    q = picks["agreement_strength"].quantile([0.33, 0.66]).to_dict()
    picks["strength_bucket"] = pd.cut(
        picks["agreement_strength"],
        [-np.inf, q[0.33], q[0.66], np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )
    picks["price_bucket"] = pd.cut(
        picks["price"], [-np.inf, -180, -130, -100, 130, 180, np.inf], include_lowest=True
    ).astype(str)
    picks["edge_bucket"] = pd.cut(picks["edge"], [0, 4, 6, 10, np.inf], include_lowest=True).astype(str)
    picks["week_bucket"] = pd.cut(picks["week"], [0, 5, 10, 14, 30], labels=["W1-5", "W6-10", "W11-14", "W15+"]).astype(str)

    detail = []
    for col in ["favorite", "home_pick", "price_bucket", "edge_bucket", "week_bucket", "strength_bucket", "pick_agree"]:
        detail.extend(segment(picks, col))
    if detail:
        print("\nSEGMENTS")
        print(pd.DataFrame(detail).sort_values(["segment", "roi"], ascending=[True, False]).to_string(index=False))

    # A consensus filter is only a candidate if it keeps a useful sample and improves
    # both ROI and drawdown without a material win-rate regression.
    candidate = (
        agree["picks"] >= 15
        and agree["roi"] > base["roi"]
        and agree["max_drawdown_pct"] >= base["max_drawdown_pct"]
        and agree["winrate"] >= base["winrate"] - 0.02
    )
    print(f"CONSENSUS_CANDIDATE={candidate}")
    print(f"DELTA_ROI_PP={agree['roi'] - base['roi']:.3f}")
    print(f"DELTA_WINRATE_PP={100 * (agree['winrate'] - base['winrate']):.3f}")
    print(f"DISAGREE_ROI={disagree['roi']:.3f}")


if __name__ == "__main__":
    main()
