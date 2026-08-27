import numpy as np
import pandas as pd

PBP_METRICS = [
    "off_epa_play", "off_success_rate", "pass_epa", "rush_epa",
    "explosive_rate", "sack_rate_allowed", "plays",
    "def_epa_allowed", "def_success_allowed", "def_explosive_allowed", "pressure_rate",
]

# Capa experimental: se agrega al Data Lake pero NO entra al modelo de producción
# mientras no demuestre mejora OOS en backtest/walk-forward.
SITUATIONAL_METRICS = [
    "early_down_epa", "early_down_success", "neutral_pass_rate",
    "redzone_epa", "redzone_success", "third_fourth_epa", "late_down_success",
    "early_down_epa_allowed", "redzone_epa_allowed", "late_down_epa_allowed",
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _mean_where(values, mask):
    x = pd.to_numeric(values.where(mask), errors="coerce")
    return x


def agregar_pbp_por_equipo_partido(pbp):
    """Agrega play-by-play real de nflverse a una fila por equipo/partido.

    No genera datos sintéticos. Solo usa jugadas ofensivas reales con EPA disponible.
    Las métricas situacionales se calculan exclusivamente con información de la jugada.
    """
    required = {"game_id", "season", "week", "posteam", "defteam", "epa"}
    if pbp is None or pbp.empty or not required.issubset(pbp.columns):
        return pd.DataFrame()

    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].isin(["REG", "POST"])].copy()

    for c in ["pass", "rush", "epa", "yards_gained", "sack", "qb_hit", "down", "qtr", "yardline_100", "score_differential"]:
        if c not in df.columns:
            df[c] = np.nan if c in {"epa", "yards_gained", "down", "qtr", "yardline_100", "score_differential"} else 0

    df["epa"] = _num(df["epa"])
    df["yards_gained"] = _num(df["yards_gained"])
    df["down"] = _num(df["down"])
    df["qtr"] = _num(df["qtr"])
    df["yardline_100"] = _num(df["yardline_100"])
    df["score_differential"] = _num(df["score_differential"])
    df["pass"] = _num(df["pass"]).fillna(0).astype(int)
    df["rush"] = _num(df["rush"]).fillna(0).astype(int)
    df["sack"] = _num(df["sack"]).fillna(0).astype(int)
    df["qb_hit"] = _num(df["qb_hit"]).fillna(0).astype(int)

    plays = df[(df["posteam"].notna()) & (df["defteam"].notna()) & df["epa"].notna() & ((df["pass"] == 1) | (df["rush"] == 1))].copy()
    if plays.empty:
        return pd.DataFrame()

    if "success" in plays.columns:
        plays["success_real"] = _num(plays["success"]).fillna((plays["epa"] > 0).astype(float))
    else:
        plays["success_real"] = (plays["epa"] > 0).astype(float)

    plays["explosive"] = (((plays["pass"] == 1) & (plays["yards_gained"] >= 20)) | ((plays["rush"] == 1) & (plays["yards_gained"] >= 10))).astype(float)
    plays["pressure_proxy"] = ((plays["sack"] == 1) | (plays["qb_hit"] == 1)).astype(float)
    plays["pass_epa_val"] = plays["epa"].where(plays["pass"] == 1)
    plays["rush_epa_val"] = plays["epa"].where(plays["rush"] == 1)
    plays["sack_allowed"] = plays["sack"].where(plays["pass"] == 1)

    early = plays["down"].isin([1, 2])
    redzone = plays["yardline_100"].between(0, 20, inclusive="both")
    late_down = plays["down"].isin([3, 4])
    # Neutral script: 1Q-3Q, early downs y marcador dentro de una posesión.
    neutral = early & plays["qtr"].between(1, 3, inclusive="both") & plays["score_differential"].between(-8, 8, inclusive="both")

    plays["early_down_epa_val"] = _mean_where(plays["epa"], early)
    plays["early_down_success_val"] = _mean_where(plays["success_real"], early)
    plays["neutral_pass_val"] = plays["pass"].where(neutral)
    plays["redzone_epa_val"] = _mean_where(plays["epa"], redzone)
    plays["redzone_success_val"] = _mean_where(plays["success_real"], redzone)
    plays["third_fourth_epa_val"] = _mean_where(plays["epa"], late_down)
    plays["late_down_success_val"] = _mean_where(plays["success_real"], late_down)

    keys = ["game_id", "season", "week", "posteam", "defteam"]
    off = plays.groupby(keys, dropna=False).agg(
        off_epa_play=("epa", "mean"),
        off_success_rate=("success_real", "mean"),
        pass_epa=("pass_epa_val", "mean"),
        rush_epa=("rush_epa_val", "mean"),
        explosive_rate=("explosive", "mean"),
        sack_rate_allowed=("sack_allowed", "mean"),
        plays=("epa", "size"),
        early_down_epa=("early_down_epa_val", "mean"),
        early_down_success=("early_down_success_val", "mean"),
        neutral_pass_rate=("neutral_pass_val", "mean"),
        redzone_epa=("redzone_epa_val", "mean"),
        redzone_success=("redzone_success_val", "mean"),
        third_fourth_epa=("third_fourth_epa_val", "mean"),
        late_down_success=("late_down_success_val", "mean"),
    ).reset_index().rename(columns={"posteam": "team", "defteam": "opponent"})

    deff = plays.groupby(keys, dropna=False).agg(
        def_epa_allowed=("epa", "mean"),
        def_success_allowed=("success_real", "mean"),
        def_explosive_allowed=("explosive", "mean"),
        pressure_rate=("pressure_proxy", "mean"),
        early_down_epa_allowed=("early_down_epa_val", "mean"),
        redzone_epa_allowed=("redzone_epa_val", "mean"),
        late_down_epa_allowed=("third_fourth_epa_val", "mean"),
    ).reset_index().rename(columns={"defteam": "team", "posteam": "opponent"})

    deff = deff[["game_id", "team", "def_epa_allowed", "def_success_allowed", "def_explosive_allowed", "pressure_rate",
                 "early_down_epa_allowed", "redzone_epa_allowed", "late_down_epa_allowed"]]
    out = off.merge(deff, on=["game_id", "team"], how="left")
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def construir_pbp_pregame(df_games, pbp_team_game, windows=(4, 8), metrics=None):
    """Construye features PBP rolling estrictamente anteriores a cada semana."""
    if pbp_team_game is None or pbp_team_game.empty:
        return pd.DataFrame()

    metrics = list(PBP_METRICS if metrics is None else metrics)
    games = df_games.copy()
    if "game_type" in games.columns:
        games = games[games["game_type"].isin(["REG", "POST"])].copy()
    game_cols = ["game_id", "season", "week", "home_team", "away_team"]
    games = games[[c for c in game_cols if c in games.columns]].dropna(subset=["game_id", "season", "week"])

    p = pbp_team_game.copy()
    p["season"] = _num(p["season"])
    p["week"] = _num(p["week"])
    p = p.dropna(subset=["game_id", "season", "week", "team"])

    history = {}
    rows = []
    for (season, week), week_games in games.sort_values(["season", "week", "game_id"]).groupby(["season", "week"], sort=True):
        for _, g in week_games.iterrows():
            row = {"game_id": g["game_id"]}
            ok = True
            for side, team in [("home", g.get("home_team")), ("away", g.get("away_team"))]:
                hist = history.get(team, [])
                if len(hist) < min(windows):
                    ok = False
                    break
                hdf = pd.DataFrame(hist)
                for metric in metrics:
                    if metric not in hdf.columns:
                        row[f"{side}_{metric}_{windows[0]}"] = np.nan
                        for w in windows[1:]:
                            row[f"{side}_{metric}_{w}"] = np.nan
                        continue
                    for w in windows:
                        row[f"{side}_{metric}_{w}"] = pd.to_numeric(hdf[metric], errors="coerce").tail(w).mean()
            if ok:
                rows.append(row)

        ids = set(week_games["game_id"].astype(str))
        week_pbp = p[p["game_id"].astype(str).isin(ids)]
        for _, r in week_pbp.iterrows():
            team = r["team"]
            history.setdefault(team, []).append({m: r.get(m) for m in metrics})
            history[team] = history[team][-16:]

    return pd.DataFrame(rows)


def features_pbp_actuales(pbp_team_game, home_team, away_team, windows=(4, 8), metrics=None):
    if pbp_team_game is None or pbp_team_game.empty:
        return None
    metrics = list(PBP_METRICS if metrics is None else metrics)
    out = {}
    for side, team in [("home", home_team), ("away", away_team)]:
        h = pbp_team_game[pbp_team_game["team"] == team].sort_values(["season", "week", "game_id"])
        if len(h) < min(windows):
            return None
        for metric in metrics:
            if metric not in h.columns:
                return None
            vals = pd.to_numeric(h[metric], errors="coerce")
            for w in windows:
                out[f"{side}_{metric}_{w}"] = vals.tail(w).mean()
    return out
