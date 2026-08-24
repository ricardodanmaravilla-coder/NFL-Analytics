import os
import pandas as pd
import nfl_data_py as nfl


def crear_directorio_data():
    os.makedirs("data", exist_ok=True)


def _columnas_existentes(df, columnas):
    return [c for c in columnas if c in df.columns]


def descargar_base_nfl():
    print("🏈 Iniciando descarga de la base de datos NFL...")
    crear_directorio_data()

    temporadas = [2021, 2022, 2023, 2024, 2025, 2026]
    print(f"📥 Descargando temporadas {temporadas}...")

    df_games = nfl.import_schedules(temporadas)

    # Histórico terminado únicamente. Nunca inventamos resultados faltantes.
    df_games = df_games[df_games["result"].notna()].copy()

    # Conservamos tanto el resultado real como las líneas/precios prepartido
    # cuando nflverse los proporciona. Los faltantes permanecen NaN: la app
    # debe marcar NO BET si no existe un precio real.
    columnas_juegos = [
        "game_id", "season", "game_type", "week", "gameday", "gametime",
        "home_team", "away_team", "home_score", "away_score", "result", "total",
        "spread_line", "total_line",
        "home_moneyline", "away_moneyline",
        "home_spread_odds", "away_spread_odds",
        "over_odds", "under_odds",
        "home_rest", "away_rest",
        "stadium", "roof", "surface", "temp", "wind"
    ]

    columnas_disponibles = _columnas_existentes(df_games, columnas_juegos)
    df_ml = df_games[columnas_disponibles].copy()

    # No rellenamos clima, líneas ni cuotas con valores ficticios.
    # total es el total REAL final; total_line es la línea prepartido.
    ruta_csv_juegos = "data/historico_nfl_games.csv"
    df_ml.to_csv(ruta_csv_juegos, index=False)
    print(f"✅ {len(df_ml)} partidos históricos guardados en {ruta_csv_juegos}")

    print("\n🎯 Descargando métricas semanales reales de QBs...")
    anios_qbs = [2022, 2023, 2024, 2025, 2026]
    df_qbs_list = []

    for anio in anios_qbs:
        try:
            print(f"   ⏳ {anio}...")
            df_temp = nfl.import_weekly_data([anio])
            if df_temp is not None and not df_temp.empty:
                df_qbs_list.append(df_temp)
                print("   ✅ OK")
        except Exception as e:
            print(f"   ⚠️ {anio} no disponible: {e}")

    if df_qbs_list:
        df_qbs = pd.concat(df_qbs_list, ignore_index=True)
        cols_qb = [
            "player_id", "player_name", "recent_team", "season", "week",
            "completions", "attempts", "passing_yards", "passing_tds",
            "interceptions", "sacks", "sack_yards",
            "passing_air_yards", "passing_yards_after_catch",
            "passing_first_downs", "passing_epa"
        ]
        cols_disponibles = _columnas_existentes(df_qbs, cols_qb)
        df_qbs_limpio = df_qbs[cols_disponibles].copy()
        ruta_csv_qbs = "data/historico_nfl_qbs.csv"
        df_qbs_limpio.to_csv(ruta_csv_qbs, index=False)
        print(f"✅ Métricas de QBs guardadas en {ruta_csv_qbs}")
    else:
        print("❌ No se pudo descargar ningún año de estadísticas de QBs.")


if __name__ == "__main__":
    descargar_base_nfl()
