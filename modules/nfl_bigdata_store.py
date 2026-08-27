from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_PBP_CSV = Path("data/historico_nfl_pbp_team_game.csv")
DEFAULT_PBP_PARQUET = Path("data/parquet/pbp_team_game")


def csv_a_parquet_pbp(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET):
    """Convierte el agregado PBP real existente a Parquet particionado por temporada."""
    csv_path = Path(csv_path)
    parquet_dir = Path(parquet_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = {"game_id", "season", "week", "team"}
    if not required.issubset(df.columns):
        raise ValueError(f"PBP inválido: faltan {sorted(required - set(df.columns))}")

    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    for season, sdf in df.groupby("season", sort=True):
        season_dir = parquet_dir / f"season={int(season)}"
        season_dir.mkdir(parents=True, exist_ok=True)
        sdf.to_parquet(season_dir / "pbp_team_game.parquet", index=False, compression="zstd")
    return parquet_dir


def leer_pbp_parquet(parquet_dir=DEFAULT_PBP_PARQUET, seasons=None):
    """Lee PBP agregado con DuckDB sin cargar archivos innecesarios."""
    parquet_dir = Path(parquet_dir)
    pattern = str(parquet_dir / "season=*" / "*.parquet")
    con = duckdb.connect(database=":memory:")
    try:
        if seasons:
            seasons = [int(x) for x in seasons]
            placeholders = ",".join("?" for _ in seasons)
            sql = f"SELECT * FROM read_parquet(?, hive_partitioning=true) WHERE season IN ({placeholders}) ORDER BY season, week, game_id, team"
            return con.execute(sql, [pattern, *seasons]).fetch_df()
        return con.execute(
            "SELECT * FROM read_parquet(?, hive_partitioning=true) ORDER BY season, week, game_id, team",
            [pattern],
        ).fetch_df()
    finally:
        con.close()


def cargar_pbp_preferente(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET):
    """Usa Parquet/DuckDB si existe; mantiene CSV como fallback seguro."""
    parquet_dir = Path(parquet_dir)
    if parquet_dir.exists() and any(parquet_dir.rglob("*.parquet")):
        return leer_pbp_parquet(parquet_dir)
    return pd.read_csv(csv_path)


def validar_paridad_csv_parquet(csv_path=DEFAULT_PBP_CSV, parquet_dir=DEFAULT_PBP_PARQUET):
    """Verifica que la migración no altere filas, claves ni métricas numéricas."""
    csv = pd.read_csv(csv_path).sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    pq = leer_pbp_parquet(parquet_dir).sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    common = [c for c in csv.columns if c in pq.columns]
    pq = pq[common]
    csv = csv[common]
    if len(csv) != len(pq):
        raise AssertionError(f"Filas distintas CSV={len(csv)} Parquet={len(pq)}")
    for key in ["game_id", "season", "week", "team"]:
        if not csv[key].astype(str).equals(pq[key].astype(str)):
            raise AssertionError(f"Clave distinta: {key}")
    num = [c for c in common if c not in {"game_id", "team", "opponent"}]
    for c in num:
        a = pd.to_numeric(csv[c], errors="coerce")
        b = pd.to_numeric(pq[c], errors="coerce")
        if not a.fillna(0).round(10).equals(b.fillna(0).round(10)):
            raise AssertionError(f"Métrica distinta: {c}")
    return {"rows": len(csv), "columns": len(common), "ok": True}
