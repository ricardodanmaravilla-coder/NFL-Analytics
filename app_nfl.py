import streamlit as st
import pandas as pd
import requests
import os
import datetime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_ml_engine import PredictorNFL_ML

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="🏈 NFL Analytics & Value Betting", layout="wide")

API_KEY = os.environ.get("API_SPORTS_KEY")
HEADERS = {'x-apisports-key': API_KEY}
BASE_URL_NFL = "https://v1.american-football.api-sports.io"

# --- COORDENADAS Y TIPO DE ESTADIO PARA EL CLIMA ---
estadios_info = {
    "BUF": {"lat": 42.773, "lon": -78.786, "dome": False},
    "MIA": {"lat": 25.957, "lon": -80.238, "dome": False},
    "NE":  {"lat": 42.090, "lon": -71.264, "dome": False},
    "NYJ": {"lat": 40.813, "lon": -74.074, "dome": False},
    "BAL": {"lat": 39.277, "lon": -76.622, "dome": False},
    "CIN": {"lat": 39.095, "lon": -84.516, "dome": False},
    "CLE": {"lat": 41.506, "lon": -81.699, "dome": False},
    "PIT": {"lat": 40.446, "lon": -80.015, "dome": False},
    "HOU": {"lat": 29.684, "lon": -95.410, "dome": True},
    "IND": {"lat": 39.760, "lon": -86.163, "dome": True},
    "JAX": {"lat": 30.323, "lon": -81.637, "dome": False},
    "TEN": {"lat": 36.166, "lon": -86.771, "dome": False},
    "DEN": {"lat": 39.743, "lon": -105.020, "dome": False},
    "KC":  {"lat": 39.048, "lon": -94.483, "dome": False},
    "LV":  {"lat": 36.090, "lon": -115.183, "dome": True},
    "LAC": {"lat": 33.953, "lon": -118.339, "dome": True},
    "DAL": {"lat": 32.747, "lon": -97.092, "dome": True},
    "NYG": {"lat": 40.813, "lon": -74.074, "dome": False},
    "PHI": {"lat": 39.900, "lon": -75.167, "dome": False},
    "WAS": {"lat": 38.907, "lon": -76.864, "dome": False},
    "CHI": {"lat": 41.862, "lon": -87.616, "dome": False},
    "DET": {"lat": 42.340, "lon": -83.045, "dome": True},
    "GB":  {"lat": 44.501, "lon": -88.062, "dome": False},
    "MIN": {"lat": 44.973, "lon": -93.257, "dome": True},
    "ATL": {"lat": 33.755, "lon": -84.400, "dome": True},
    "CAR": {"lat": 35.225, "lon": -80.852, "dome": False},
    "NO":  {"lat": 29.951, "lon": -90.081, "dome": True},
    "TB":  {"lat": 27.975, "lon": -82.503, "dome": False},
    "ARI": {"lat": 33.527, "lon": -112.262, "dome": True},
    "LA":  {"lat": 33.953, "lon": -118.339, "dome": True},
    "SF":  {"lat": 37.403, "lon": -121.969, "dome": False},
    "SEA": {"lat": 47.595, "lon": -122.331, "dome": False}
}

st.title("🏈 NFL Analytics & Value Betting (Automático)")

# --- CARGAR HISTÓRICO ---
@st.cache_data(ttl=3600)
def cargar_datos_nfl():
    try:
        df_games = pd.read_csv('data/historico_nfl_games.csv')
        df_qbs = pd.read_csv('data/historico_nfl_qbs.csv')
        return df_games, df_qbs
    except:
        st.error("No se encontraron los CSV. Corre el GitHub Action primero.")
        return pd.DataFrame(), pd.DataFrame()

df_games, df_qbs = cargar_datos_nfl()

# --- FUNCIONES AUTOMÁTICAS ---
def obtener_clima_estadio(equipo_local):
    """Consulta el clima en vivo usando Open-Meteo (Gratis)"""
    info = estadios_info.get(equipo_local)
    if not info: return 70.0, 0.0, False
    if info["dome"]: return 70.0, 0.0, True
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={info['lat']}&longitude={info['lon']}&current_weather=true"
        res = requests.get(url).json()
        temp_c = res["current_weather"]["temperature"]
        temp_f = (temp_c * 9/5) + 32  # Fahrenheit para el modelo
        wind_kmh = res["current_weather"]["windspeed"]
        wind_mph = wind_kmh * 0.621371 # Millas por hora (MPH)
        return round(temp_f, 1), round(wind_mph, 1), False
    except:
        return 65.0, 5.0, False

def obtener_calendario_nfl(temporada, semana):
    """Obtiene los partidos de la semana directamente de la base de datos oficial"""
    try:
        juegos_temp = df_games[(df_games['season'] == temporada) & (df_games['week'] == semana)].copy()
        
        if juegos_temp.empty:
            import nfl_data_py as nfl
            df_sched = nfl.import_schedules([temporada])
            juegos_temp = df_sched[(df_sched['season'] == temporada) & (df_sched['week'] == semana)].copy()
            
        return juegos_temp
    except Exception as e:
        st.error(f"Error al obtener el calendario: {e}")
        return pd.DataFrame()

# --- PANEL AUTOMÁTICO DE JORNADA ---
st.markdown("### 🤖 Escáner Automático de Jornada (NFL)")

col_auto1, col_auto2, col_auto3 = st.columns(3)
with col_auto1:
    temporada_auto = st.number_input("Temporada:", min_value=2020, max_value=2030, value=2026)
with col_auto2:
    semana_auto = st.number_input("Semana a escanear:", min_value=1, max_value=22, value=1)
with col_auto3:
    min_prob_filtro = st.slider("Filtro de Probabilidad Mínima (%):", min_value=50, max_value=80, value=60, step=1)

if st.button("🚀 Escanear con Filtro Estricto (60%+)", type="primary"):
    with st.spinner("Filtrando oportunidades de oro y cruzando modelos..."):
        juegos_df = obtener_calendario_nfl(temporada_auto, semana_auto)
        
        if juegos_df.empty:
            st.warning(f"⚠️ No se encontraron partidos para la Temporada {temporada_auto}, Semana {semana_auto}.")
        else:
            ml_engine = PredictorNFL_ML()
            ml_engine.entrenar(df_games, df_qbs)
            
            oportunidades_encontradas = 0
            
            st.write("---")
            st.write(f"🔎 **Escaneo con filtro del {min_prob_filtro}%+ para la Semana {semana_auto}:**")
            
            for _, j in juegos_df.iterrows():
                home_code = j.get("home_team")
                away_code = j.get("away_team")
                fecha_partido = str(j.get("gameday", "Fecha por confirmar"))
                
                if not home_code or not away_code:
                    continue
                    
                # Clima automático
                temp, wind, is_dome = obtener_clima_estadio(home_code)
                
                # Líneas de Las Vegas
                linea_ou_api = float(j.get("total", 45.5)) if pd.notna(j.get("total")) else 45.5
                spread_api = float(j.get("spread_line", -3.0)) if pd.notna(j.get("spread_line")) else -3.0
                
                # Ejecutar motores
                mc = simular_nfl_montecarlo(home_code, away_code, df_games, linea_ou_api, spread_api)
                ml = ml_engine.predecir_contexto(semana_auto, home_code, temp, wind, 1 if is_dome else 0)
                
                total_mc = mc['Proyeccion_Score']['Total_Proyectado']
                total_ml = ml['ML_Puntos_Totales_Esperados']
                
                prob_over = mc['Over_Under']['Prob Over']
                prob_under = mc['Over_Under']['Prob Under']
                
                # Lógica de Consenso EV+ aplicando tu filtro estricto (> min_prob_filtro)
                es_over = (total_mc > linea_ou_api) and (total_ml > linea_ou_api) and (prob_over >= min_prob_filtro)
                es_under = (total_mc < linea_ou_api) and (total_ml < linea_ou_api) and (prob_under >= min_prob_filtro)
                
                clima_str = "🏠 Domo" if is_dome else f"🌡️ {temp}°F 💨 {wind}mph"
                
                # Solo mostrar si pasa el filtro o mostrar todas con etiqueta de advertencia
                with st.expander(f"📅 {fecha_partido} | 🏈 {away_code} @ {home_code} | Línea O/U: {linea_ou_api}"):
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.markdown("**📊 Métricas Base (Montecarlo)**")
                        st.write(f"- Proyección Score: {away_code} **{mc['Proyeccion_Score'][away_code]}** - **{mc['Proyeccion_Score'][home_code]}** {home_code}")
                        st.write(f"- Total Proyectado: **{total_mc} pts**")
                        st.write(f"- Prob. Over: **{prob_over}%**")
                        st.write(f"- Prob. Under: **{prob_under}%**")
                        
                    with col_det2:
                        st.markdown("**🤖 Ajuste de Entorno (Machine Learning)**")
                        st.write(f"- Clima del Estadio: {clima_str}")
                        st.write(f"- Puntos Ajustados por IA: **{total_ml} pts**")
                        st.write(f"- Margen Local Previsto: **{ml['ML_Margen_Local_Esperado']} pts**")
                        
                        # Nota sobre QBs
                        if not df_qbs.empty:
                            st.write("- 🏈 *Datos de QBs:* Base histórica cargada (Pendiente de integrar métrica de yardas finas).")
                    
                    st.markdown("---")
                    
                    # Veredicto final con el filtro estricto aplicado
                    if es_over:
                        oportunidades_encontradas += 1
                        st.success(f"🎯 **APUESTA RECOMENDADA (OVER {linea_ou_api}):** Consenso de modelos con **{prob_over}%** de efectividad (Supera tu filtro del {min_prob_filtro}%).")
                    elif es_under:
                        oportunidades_encontradas += 1
                        st.success(f"🎯 **APUESTA RECOMENDADA (UNDER {linea_ou_api}):** Consenso de modelos con **{prob_under}%** de efectividad (Supera tu filtro del {min_prob_filtro}%).")
                    else:
                        st.warning(f"⚠️ **DESCARTADO:** No cumple con el filtro mínimo del {min_prob_filtro}% de efectividad o los motores discrepan.")

            st.write("---")
            st.success(f"🎯 **Escaneo finalizado.** Se encontraron **{oportunidades_encontradas} apuestas de alto valor** que superan tu filtro del {min_prob_filtro}%.")
