"""Production Cloud Run entrypoint with Parquet-first historical PBP loading.

`cloudrun_api` owns the HTTP routes and betting logic. This wrapper replaces only
its historical-data loader before the first request is served, so production uses
the partitioned Parquet lake when present and retains CSV as a safe fallback.
"""
from functools import lru_cache

import pandas as pd

import cloudrun_api as base
from modules.nfl_bigdata_store import cargar_pbp_preferente, detectar_storage_pbp


@lru_cache(maxsize=1)
def load_history_parquet_first():
    games = pd.read_csv("data/historico_nfl_games.csv")
    pbp = cargar_pbp_preferente()
    return games, pbp


# Patch the module global referenced dynamically by get_models(). Importing
# cloudrun_api does not train a model or load history, so this happens before use.
base.load_history = load_history_parquet_first
base.MODEL_CACHE.clear()
base.PBP_STORAGE = detectar_storage_pbp()

app = base.app
