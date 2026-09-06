"""Fresh production history loader for NFL Analytics.

Cloud Run images contain a validated local snapshot for cold-start resilience, but a
long-lived revision must not depend on that snapshot forever. The preferred runtime
path reads the latest completed-games CSV and season-partitioned aggregate PBP
Parquet from this repository's `main` branch. Every remote payload is schema-checked;
any network/schema failure falls back atomically to the bundled local data.
"""
from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from modules.nfl_bigdata_store import cargar_pbp_preferente

REMOTE_ROOT = "https://raw.githubusercontent.com/ricardodanmaravilla-coder/NFL-Analytics/main"
LOCAL_GAMES = Path("data/historico_nfl_games.csv")

GAME_REQUIRED = {
    "game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"
}
PBP_REQUIRED = {
    "game_id", "season", "week", "team", "off_epa_play", "off_success_rate",
    "def_epa_allowed",
}


def _validate(df: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{label} vacío")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} sin columnas requeridas: {sorted(missing)}")
    return df


def _get(url: str, timeout: float = 12.0) -> requests.Response:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def _remote_games(timeout: float = 12.0) -> pd.DataFrame:
    response = _get(f"{REMOTE_ROOT}/data/historico_nfl_games.csv", timeout=timeout)
    games = pd.read_csv(StringIO(response.text))
    return _validate(games, GAME_REQUIRED, "games remoto")


def _remote_pbp_for_seasons(seasons: Iterable[int], timeout: float = 12.0) -> pd.DataFrame:
    frames = []
    for season in sorted({int(x) for x in seasons}):
        url = (
            f"{REMOTE_ROOT}/data/parquet/pbp_team_game/"
            f"season={season}/pbp_team_game.parquet"
        )
        response = _get(url, timeout=timeout)
        frame = pd.read_parquet(BytesIO(response.content))
        frame = _validate(frame, PBP_REQUIRED, f"PBP remoto {season}")
        frames.append(frame)
    if not frames:
        raise ValueError("Sin temporadas PBP remotas")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def _local_history() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    games = _validate(pd.read_csv(LOCAL_GAMES), GAME_REQUIRED, "games local")
    pbp = _validate(cargar_pbp_preferente(), PBP_REQUIRED, "PBP local")
    return games, pbp, "LOCAL_PARQUET_OR_CSV_FALLBACK"


def load_production_history(prefer_remote: bool = True, timeout: float = 12.0):
    """Return `(games, pbp, source)` with atomic remote->local fallback.

    Only seasons actually present in completed historical games are requested. This
    matters before Week 1, when the current season may legitimately have no PBP
    partition yet. Callers still apply `historico_antes()` for their target week.
    """
    if prefer_remote:
        try:
            games = _remote_games(timeout=timeout)
            seasons = pd.to_numeric(games["season"], errors="coerce").dropna().astype(int).unique()
            pbp = _remote_pbp_for_seasons(seasons, timeout=timeout)
            # Require every PBP row to correspond to a known completed-game id. This
            # catches accidental mixed datasets without discarding valid team rows.
            game_ids = set(games["game_id"].dropna().astype(str))
            if not set(pbp["game_id"].dropna().astype(str)).issubset(game_ids):
                raise ValueError("PBP remoto contiene game_id fuera del histórico remoto")
            return games, pbp, "REMOTE_PARQUET"
        except Exception:
            pass
    return _local_history()
