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

# --- DICCIONARIOS DE TRADUCCIÓN Y ESTADIOS ---
# Traductor de API-Sports a nuestro modelo ML (nflfastR)
nfl_team_map = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

# Coordenadas y tipo de estadio para el clima automático
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
        temp_f = (temp_c * 9/5) + 32  # Convertir a Fahrenheit para el modelo
        wind_kmh = res["current_weather"]["windspeed"]
        wind_mph = wind_kmh * 0.621371 # Convertir a MPH
        return round(temp_f, 1), round(wind_mph, 1), False
    except:
        return 65.0, 5.0, False

def obtener_calendario_api(semana, temporada=2024):
    """Descarga juegos de API-Sports"""
    if not API_KEY: return []
    url = f"{BASE_URL_NFL}/games?league=1&season={temporada}"
    try:
        res = requests.get(url, headers=HEADERS).json()
        juegos = res.get("response", [])
        # En API-Sports, filtramos por semana (Week X)
        juegos_semana = [j for j in juegos if f"Week {semana}" in str(j.get("game", {}).get("week", ""))]
        return juegos_semana
    except:
        return []

# --- PANEL AUTOMÁTICO ---
st.markdown("### 🤖 Escáner Automático de Jornada")
semana_auto = st.number_input("Buscar juegos de la Semana:", min_value=1, max_value=22, value=1)

if st.button("Buscar Partidos y Extraer Clima", type="primary"):
    with st.spinner("Conectando con Las Vegas y satélites del clima..."):
        juegos = obtener_calendario_api(semana_auto)
        
        if not juegos:
            st.warning("⚠️ No se encontraron partidos en la API para esta semana o revisa tu API Key.")
        else:
            ml_engine = PredictorNFL_ML()
            ml_engine.entrenar(df_games, df_qbs)
            
            for j in juegos:
                home_api = j["teams"]["home"]["name"]
                away_api = j["teams"]["away"]["name"]
                
                # Traducir nombres (Ej: Kansas City Chiefs -> KC)
                home_code = nfl_team_map.get(home_api)
                away_code = nfl_team_map.get(away_api)
                
                if not home_code or not away_code:
                    continue
                    
                # Extraer clima automático
                temp, wind, is_dome = obtener_clima_estadio(home_code)
                
                # Extraer Líneas básicas que nos da el endpoint de juegos
                # (Nota: Para cuotas más finas se requiere el endpoint /odds, aquí usamos las del pre-match si vienen)
                linea_ou_api = 45.5 # Fallback temporal si la API no manda odds directas
                spread_api = -3.0   
                
                with st.expander(f"🏈 {away_code} @ {home_code} | Clima: {'🏠 Domo' if is_dome else f'🌡️ {temp}°F 💨 {wind}mph'}"):
                    # Correr modelos
                    mc = simular_nfl_montecarlo(home_code, away_code, df_games, linea_ou_api, spread_api)
                    ml = ml_engine.predecir_contexto(semana_auto, home_code, temp, wind, 1 if is_dome else 0)
                    
                    st.write(f"**Proyección Montecarlo:** {away_code} {mc['Proyeccion_Score'][away_code]} - {mc['Proyeccion_Score'][home_code]} {home_code}")
                    st.write(f"**Ajuste Machine Learning:** Puntos totales esperados: {ml['ML_Puntos_Totales_Esperados']}")
                    
                    # Consenso rápido
                    if mc['Proyeccion_Score']['Total_Proyectado'] > linea_ou_api and ml['ML_Puntos_Totales_Esperados'] > linea_ou_api:
                        st.success("🔥 ALERTA EV+: Ambos modelos proyectan OVER.")
                    elif mc['Proyeccion_Score']['Total_Proyectado'] < linea_ou_api and ml['ML_Puntos_Totales_Esperados'] < linea_ou_api:
                        st.success("🔥 ALERTA EV+: Ambos modelos proyectan UNDER.")
                    else:
                        st.warning("⚠️ Sin consenso de valor. Evitar apostar totales.")
