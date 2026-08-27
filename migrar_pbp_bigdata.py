import argparse

from modules.nfl_bigdata_store import construir_lake_nflverse, csv_a_parquet_pbp, validar_paridad_csv_parquet


def main():
    parser = argparse.ArgumentParser(description="Construye/valida el Data Lake PBP NFL")
    parser.add_argument("--download", action="store_true", help="Descarga PBP real de nflverse antes de validar")
    parser.add_argument("--start", type=int, default=2021)
    parser.add_argument("--end", type=int, default=2025, help="Por defecto excluye 2026 del dataset de desarrollo")
    args = parser.parse_args()
    if args.download:
        if args.end < args.start:
            raise SystemExit("--end debe ser >= --start")
        print(f"✅ Data Lake nflverse: {construir_lake_nflverse(range(args.start, args.end + 1))}")
    else:
        print(f"✅ Parquet PBP creado en {csv_a_parquet_pbp()}")
    print(f"✅ Paridad CSV↔Parquet: {validar_paridad_csv_parquet()}")


if __name__ == "__main__":
    main()
