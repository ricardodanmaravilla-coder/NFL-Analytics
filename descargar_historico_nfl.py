import pandas as pd
import nfl_data_py as nfl
import os

def crear_directorio_data():
    if not os.path.exists('data'):
        os.makedirs('data')

def descargar_base_nfl():
    print("🏈 Iniciando descarga de la base de datos de la NFL...")
    crear_directorio_data()
    
    # 1. Descargar el calendario y resultados históricos (Últimos años)
    temporadas = [2021, 2022, 2023, 2024, 2025, 2026]
    print(f"📥 Descargando temporadas {temporadas}...")
    
    df_games = nfl.import_schedules(temporadas)
    df_games = df_games[~df_games['result'].isna()].copy()
    
    columnas_clave = [
        'game_id', 'season', 'game_type', 'week', 'gameday',
        'home_team', 'away_team', 'home_score', 'away_score', 'result', 'total', 
        'stadium', 'roof', 'surface', 'temp', 'wind'
    ]
    
    df_ml = df_games[columnas_clave].copy()
    df_ml['temp'] = df_ml['temp'].fillna(70)
    df_ml['wind'] = df_ml['wind'].fillna(0) 
    
    ruta_csv_juegos = 'data/historico_nfl_games.csv'
    df_ml.to_csv(ruta_csv_juegos, index=False)
    
    print(f"✅ ¡Éxito! Se descargaron {len(df_ml)} partidos históricos.")
    print(f"📁 Guardado en: {ruta_csv_juegos}")
    
    # 2. Descargar estadísticas avanzadas de Quarterbacks a prueba de errores
    print("\n🎯 Descargando métricas de Quarterbacks (Yardas, Pases)...")
    
    anios_qbs = [2022, 2023, 2024, 2025, 2026]
    df_qbs_list = []
    
    for anio in anios_qbs:
        try:
            print(f"   ⏳ Buscando stats de QBs del {anio}...")
            df_temp = nfl.import_weekly_data([anio])
            df_qbs_list.append(df_temp)
            print(f"   ✅ Stats de {anio} descargadas con éxito.")
        except Exception as e:
            print(f"   ⚠️ El año {anio} aún no está disponible en la base de datos (Error 404). Saltando...")
            
    if df_qbs_list:
        df_qbs = pd.concat(df_qbs_list, ignore_index=True)
        cols_qb = ['player_name', 'recent_team', 'season', 'week', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks']
        
        # Nos aseguramos de que las columnas existan antes de filtrar
        cols_disponibles = [col for col in cols_qb if col in df_qbs.columns]
        df_qbs_limpio = df_qbs[cols_disponibles].copy()
        
        ruta_csv_qbs = 'data/historico_nfl_qbs.csv'
        df_qbs_limpio.to_csv(ruta_csv_qbs, index=False)
        print(f"\n🎉 ¡Todo listo! Métricas de QBs guardadas en: {ruta_csv_qbs}")
    else:
        print("\n❌ Error Crítico: No se pudo descargar ningún año de estadísticas de QBs.")

if __name__ == "__main__":
    descargar_base_nfl()
