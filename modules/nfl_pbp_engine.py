import numpy as np
import pandas as pd

PBP_METRICS = [
    "off_epa_play", "off_success_rate", "pass_epa", "rush_epa",
    "explosive_rate", "sack_rate_allowed", "plays",
    "def_epa_allowed", "def_success_allowed", "def_explosive_allowed", "pressure_rate",
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def agregar_pbp_por_equipo_partido(pbp):
    """Agrega play-by-play real de nflverse a una fila por equipo/partido.

    No genera datos sintéticos. Solo usa jugadas ofensivas reales con EPA disponible.
    """
    required = {"game_id", "season", "week", "posteam", "defteam", "epa"}
    if pbp is None or pbp.empty or not required.issubset(pbp.columns):
        return pd.DataFrame()

    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].isin(["REG", "POST"])].copy()

    for c in ["pass", "rush", "epa", "yards_gained", "sack", "qb_hit"]:
        if c not in df.columns:
            df[c] = np.nan if c in {"epa", "yards_gained"} else 0

    df["epa"] = _num(df["epa"])
    df["yards_gained"] = _num(df["yards_gained"])
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

    keys = ["game_id", "season", "week", "posteam", "defteam"]
    off = plays.groupby(keys, dropna=False).agg(
        off_epa_play=("epa", "mean"),
        off_success_rate=("success_real", "mean"),
        pass_epa=("pass_epa_val", "mean"),
        rush_epa=("rush_epa_val", "mean"),
        explosive_rate=("explosive", "mean"),
        sack_rate_allowed=("sack_allowed", "mean"),
        plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team", "defteam": "opponent"})

    deff = plays.groupby(keys, dropna=False).agg(
        def_epa_allowed=("epa", "mean"),
        def_success_allowed=("success_real", "mean"),
        def_explosive_allowed=("explosive", "mean"),
        pressure_rate=("pressure_proxy", "mean"),
    ).reset_index().rename(columns={"defteam": "team", "posteam": "opponent"})

    deff = deff[["game_id", "team", "def_epa_allowed", "def_success_allowed", "def_explosive_allowed", "pressure_rate"]]
    out = off.merge(deff, on=["game_id", "team"], how="left")
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def construir_pbp_pregame(df_games, pbp_team_game, windows=(4, 8)):
    """Construye features PBP rolling estrictamente anteriores a cada semana."""
    if pbp_team_game is None or pbp_team_game.empty:
        return pd.DataFrame()

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
                for metric in PBP_METRICS:
                    for w in windows:
                        row[f"{side}_{metric}_{w}"] = pd.to_numeric(hdf[metric], errors="coerce").tail(w).mean()
            if ok:
                rows.append(row)

        # Importante: la semana completa se añade DESPUÉS de construir todas sus filas.
        ids = set(week_games["game_id"].astype(str))
        week_pbp = p[p["game_id"].astype(str).isin(ids)]
        for _, r in week_pbp.iterrows():
            team = r["team"]
            history.setdefault(team, []).append({m: r.get(m) for m in PBP_METRICS})
            history[team] = history[team][-16:]

    return pd.DataFrame(rows)


def features_pbp_actuales(pbp_team_game, home_team, away_team, windows=(4, 8)):
    if pbp_team_game is None or pbp_team_game.empty:
        return None
    out = {}
    for side, team in [("home", home_team), ("away", away_team)]:
        h = pbp_team_game[pbp_team_game["team"] == team].sort_values(["season", "week", "game_id"])
        if len(h) < min(windows):
            return None
        for metric in PBP_METRICS:
            vals = pd.to_numeric(h[metric], errors="coerce")
            for w in windows:
                out[f"{side}_{metric}_{w}"] = vals.tail(w).mean()
    return out
