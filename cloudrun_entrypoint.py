"""Production Cloud Run entrypoint with fresh Parquet-first history.

The HTTP routes remain in `cloudrun_api`. This wrapper patches only data/model
loading before the first request so a long-lived Cloud Run revision can consume
new weekly history without requiring a redeploy. Remote validated Parquet is
preferred; bundled local Parquet/CSV remains an atomic fallback.
"""
from functools import lru_cache
import os
import threading
import time

import pandas as pd

import cloudrun_api as base
from modules.nfl_calibration import historico_antes
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_production_history import load_production_history
from modules.nfl_therundown_odds import diagnose_date

HISTORY_TTL_SECONDS = max(900, int(os.getenv("NFL_HISTORY_TTL_SECONDS", "3600")))
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = {}
_LAST_BUCKET = None


def _bucket() -> int:
    return int(time.time() // HISTORY_TTL_SECONDS)


@lru_cache(maxsize=2)
def _history_for_bucket(bucket: int):
    games, pbp, source = load_production_history(prefer_remote=True)
    base.PBP_STORAGE = source
    return games, pbp


def load_history_parquet_first():
    games, pbp = _history_for_bucket(_bucket())
    return games.copy(), pbp.copy()


def get_models_fresh(season, week):
    """Cache trained models only inside the same history freshness bucket."""
    global _LAST_BUCKET
    bucket = _bucket()
    key = (int(season), int(week), bucket)
    with _CACHE_LOCK:
        if _LAST_BUCKET != bucket:
            _MODEL_CACHE.clear()
            _history_for_bucket.cache_clear()
            _LAST_BUCKET = bucket
        cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    games, pbp = load_history_parquet_first()
    past_games = historico_antes(games, season, week)
    past_pbp = historico_antes(pbp, season, week) if not pbp.empty else pd.DataFrame()
    ml = MoneylineRuntime()
    if not ml.entrenar(past_games, past_pbp):
        raise RuntimeError("No hay histórico suficiente para entrenar Moneyline")
    elo = MotorELONFL()
    elo.actualizar_ratings(past_games)
    result = (ml, elo, past_games)
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = result
    return result


base.load_history = load_history_parquet_first
base.get_models = get_models_fresh
base.MODEL_CACHE.clear()
base.PBP_STORAGE = "UNINITIALIZED"
base.HISTORY_TTL_SECONDS = HISTORY_TTL_SECONDS

app = base.app


@app.get("/api/storage")
def storage_status():
    games, pbp = load_history_parquet_first()
    seasons = sorted(pd.to_numeric(games["season"], errors="coerce").dropna().astype(int).unique().tolist())
    return {
        "history_source": base.PBP_STORAGE,
        "history_ttl_seconds": HISTORY_TTL_SECONDS,
        "games_rows": int(len(games)),
        "pbp_rows": int(len(pbp)),
        "seasons": seasons,
    }


@app.get("/api/odds/diagnostics/{gameday}")
def odds_diagnostics(gameday: str):
    """Inspect TheRundown availability without ever exposing the API key."""
    return diagnose_date(gameday)
