import pandas as pd
import nfl_data_py as nfl
import os

def crear_directorio_data():
    if not os.path.exists('data'):
        os.makedirs('data')

def descargar_base_nfl():
    print("🏈 Iniciando descarga de la base de datos de la NFL...")
    crear_directorio_data()
    
    # 1. Descargar el calendario y resultados históricos (Últimos 5 años)
    temporadas = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    print(f"📥 Descargando temporadas {temporadas}...")
    
    df_games = nfl.import_schedules(temporadas)
    
    # Filtrar solo juegos de temporada regular y playoffs que ya terminaron
    df_games = df_games[~df_games['result'].isna()].copy()
    
    # 2. Seleccionar y limpiar las columnas que alimentarán nuestro Machine Learning
    columnas_clave = [
        'game_id', 'season', 'game_type', 'week', 'gameday',
        'home_team', 'away_team', 'home_score', 'away_score', 'result', 'total', 
        'stadium', 'roof', 'surface', 'temp', 'wind'
    ]
    
    df_ml = df_games[columnas_clave].copy()
    
    # Limpieza básica de datos de clima para el modelo (llenar vacíos en domos)
    df_ml['temp'] = df_ml['temp'].fillna(70) # Si es techado, temperatura controlada ~70F
    df_ml['wind'] = df_ml['wind'].fillna(0)  # Si es techado, sin viento
    
    # 3. Guardar el archivo maestro de partidos
    ruta_csv_juegos = 'data/historico_nfl_games.csv'
    df_ml.to_csv(ruta_csv_juegos, index=False)
    
    print(f"✅ ¡Éxito! Se descargaron {len(df_ml)} partidos históricos con datos de clima y estadios.")
    print(f"📁 Guardado en: {ruta_csv_juegos}")
    
    # 4. Descargar estadísticas avanzadas de Quarterbacks (Último año para test)
    print("\n🎯 Descargando métricas de Quarterbacks (Yardas, Pases)...")
    df_qbs = nfl.import_weekly_data([2023, 2024, 2025])
    
    # Filtrar columnas relevantes para las props de jugadores (Yardas)
    cols_qb = ['player_name', 'recent_team', 'season', 'week', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks']
    df_qbs_limpio = df_qbs[cols_qb].copy()
    
    ruta_csv_qbs = 'data/historico_nfl_qbs.csv'
    df_qbs_limpio.to_csv(ruta_csv_qbs, index=False)
    print(f"✅ Métricas de QBs guardadas en: {ruta_csv_qbs}")

if __name__ == "__main__":
    descargar_base_nfl()
