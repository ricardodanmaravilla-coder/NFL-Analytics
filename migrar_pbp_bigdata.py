from modules.nfl_bigdata_store import csv_a_parquet_pbp, validar_paridad_csv_parquet


def main():
    path = csv_a_parquet_pbp()
    result = validar_paridad_csv_parquet(parquet_dir=path)
    print(f"✅ Parquet PBP creado en {path}")
    print(f"✅ Paridad CSV↔Parquet: {result}")


if __name__ == "__main__":
    main()
