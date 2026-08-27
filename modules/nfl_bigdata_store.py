from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from modules.nfl_pbp_engine import agregar_pbp_por_equipo_partido

DEFAULT_PBP_CSV = Path("data/historico_nfl_pbp_team_game.csv")
DEFAULT_PBP_PARQUET = Path("data/parquet/pbp_team_game")
DEFAULT_RAW_PARQUET = Path("data/parquet/pbp_raw")


def _write_partitioned(df, parquet_dir, filename):
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for season, sdf in df.groupby("season", sort=True):
        season_dir = parquet_dir / f"season={int(season)}"
        season_dir.mkdir(parents=True, exist_ok=True)
        sdf.to_parquet(season_dir / filename, index=False, compression="zstd")
    return parquet_dir


def csv_a_parquet_pbp(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"game_id", "season", "week", "team"}
    if not required.issubset(df.columns):
        raise ValueError(f"PBP inválido: faltan {sorted(required - set(df.columns))}")
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    return _write_partitioned(df, parquet_dir, "pbp_team_game.parquet")


def construir_lake_nflverse(seasons, raw_dir=DEFAULT_RAW_PARQUET, team_dir=DEFAULT_PBP_PARQUET, csv_backup=DEFAULT_PBP_CSV):
    """Descarga PBP real de nflverse, guarda raw Parquet y genera agregado equipo/partido.

    Se procesa temporada por temporada para limitar RAM. No usa 2026 para tuning por sí mismo;
    el consumidor decide qué temporadas entran en entrenamiento/validación.
    """
    try:
        import nfl_data_py as nfl
    except ImportError as exc:
        raise RuntimeError("nfl_data_py no está instalado") from exc

    raw_dir = Path(raw_dir)
    team_dir = Path(team_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    team_dir.mkdir(parents=True, exist_ok=True)
    aggregates = []
    for season in sorted({int(s) for s in seasons}):
        pbp = nfl.import_pbp_data([season], downcast=True, cache=False, include_participation=False)
        if pbp is None or pbp.empty:
            raise RuntimeError(f"nflverse no devolvió PBP para {season}")
        if "season" not in pbp.columns:
            pbp["season"] = season
        season_raw = raw_dir / f"season={season}"
        season_raw.mkdir(parents=True, exist_ok=True)
        pbp.to_parquet(season_raw / "pbp.parquet", index=False, compression="zstd")
        agg = agregar_pbp_por_equipo_partido(pbp)
        if agg.empty:
            raise RuntimeError(f"No se pudo agregar PBP de {season}")
        aggregates.append(agg)
        season_team = team_dir / f"season={season}"
        season_team.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(season_team / "pbp_team_game.parquet", index=False, compression="zstd")

    out = pd.concat(aggregates, ignore_index=True).sort_values(["season", "week", "game_id", "team"])
    csv_backup = Path(csv_backup)
    csv_backup.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_backup, index=False)
    return {"seasons": sorted(out["season"].astype(int).unique().tolist()), "rows": len(out), "raw_dir": str(raw_dir), "team_dir": str(team_dir)}


def leer_pbp_parquet(parquet_dir=DEFAULT_PBP_PARQUET, seasons=None):
    parquet_dir = Path(parquet_dir)
    pattern = str(parquet_dir / "season=*" / "*.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        if seasons:
            seasons = [int(x) for x in seasons]
            placeholders = ",".join("?" for _ in seasons)
            sql = f"SELECT * FROM read_parquet(?, hive_partitioning=true) WHERE season IN ({placeholders}) ORDER BY season, week, game_id, team"
            return con.execute(sql, [pattern, *seasons]).fetch_df()
        return con.execute("SELECT * FROM read_parquet(?, hive_partitioning=true) ORDER BY season, week, game_id, team", [pattern]).fetch_df()
    finally:
        con.close()


def cargar_pbp_preferente(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET, seasons=None):
    parquet_dir = Path(parquet_dir)
    if parquet_dir.exists() and any(parquet_dir.rglob("*.parquet")):
        return leer_pbp_parquet(parquet_dir, seasons=seasons)
    df = pd.read_csv(csv_path)
    if seasons is not None:
        df = df[pd.to_numeric(df["season"], errors="coerce").isin([int(s) for s in seasons])]
    return df


def validar_paridad_csv_parquet(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET, rtol=1e-6, atol=1e-8):
    """Valida que CSV y Parquet representen los mismos datos.

    CSV serializa floats a texto y al leerlos puede introducir diferencias binarias
    minúsculas frente al Parquet. Por eso la paridad numérica usa tolerancias estrictas,
    mientras claves y columnas no numéricas siguen exigiendo igualdad exacta.
    """
    csv = pd.read_csv(csv_path).sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    pq = leer_pbp_parquet(parquet_dir).sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    common = [c for c in csv.columns if c in pq.columns]
    pq, csv = pq[common], csv[common]
    if len(csv) != len(pq):
        raise AssertionError(f"Filas distintas CSV={len(csv)} Parquet={len(pq)}")
    for key in ["game_id", "season", "week", "team"]:
        if not csv[key].astype(str).equals(pq[key].astype(str)):
            raise AssertionError(f"Clave distinta: {key}")

    max_abs_diff = 0.0
    numeric_checked = 0
    for c in common:
        if c in {"game_id", "team", "opponent"}:
            continue
        a = pd.to_numeric(csv[c], errors="coerce")
        b = pd.to_numeric(pq[c], errors="coerce")
        # Sólo tratamos como numérica una columna cuando ambos lados son realmente
        # numéricos (ignorando nulos). Las demás se comparan como texto abajo.
        if a.notna().sum() == csv[c].notna().sum() and b.notna().sum() == pq[c].notna().sum():
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            if not np.allclose(av, bv, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.abs(av - bv)
                finite = diff[np.isfinite(diff)]
                observed = float(finite.max()) if finite.size else float("inf")
                raise AssertionError(f"Métrica distinta: {c}; max_abs_diff={observed:.12g}")
            diff = np.abs(av - bv)
            finite = diff[np.isfinite(diff)]
            if finite.size:
                max_abs_diff = max(max_abs_diff, float(finite.max()))
            numeric_checked += 1
        else:
            left = csv[c].fillna("<NA>").astype(str)
            right = pq[c].fillna("<NA>").astype(str)
            if not left.equals(right):
                raise AssertionError(f"Columna distinta: {c}")

    return {
        "rows": len(csv),
        "columns": len(common),
        "numeric_checked": numeric_checked,
        "max_abs_diff": max_abs_diff,
        "rtol": rtol,
        "atol": atol,
        "ok": True,
    }
