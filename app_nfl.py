import streamlit as st
import pandas as pd
import requests
import os
import datetime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_qb_engine import PredictorYardasQB

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="🏈 NFL Analytics & Value Betting (Sistema Pro)", layout="wide")

API_KEY = os.environ.get("API_SPORTS_KEY")
HEADERS = {'x-apisports-key': API_KEY}
BASE_URL_NFL = "https://v1.american-football.api-sports.io"

# --- COORDENADAS Y TIPO DE ESTADIO PARA EL CLIMA (BLINDADO) ---
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
    "OAK": {"lat": 36.090, "lon": -115.183, "dome": True},
    "LAC": {"lat": 33.953, "lon": -118.339, "dome": True},
    "SD":  {"lat": 33.953, "lon": -118.339, "dome": True},
    "DAL": {"lat": 32.747, "lon": -97.092, "dome": True},
    "NYG": {"lat": 40.813, "lon": -74.074, "dome": False},
    "PHI": {"lat": 39.900, "lon": -75.167, "dome": False},
    "WAS": {"lat": 38.907, "lon": -76.864, "dome": False},
    "WSH": {"lat": 38.907, "lon": -76.864, "dome": False},
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
    "LAR": {"lat": 33.953, "lon": -118.339, "dome": True},
    "SF":  {"lat": 37.403, "lon": -121.969, "dome": False},
    "SEA": {"lat": 47.595, "lon": -122.331, "dome": False}
}

st.title("🏈 NFL Analytics & Value Betting System Pro")

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

# --- INICIALIZAR MOTORES GLOBALES ---
@st.cache_resource
def calcular_elo_global(df):
    motor_elo = MotorELONFL()
    ratings = motor_elo.actualizar_ratings(df)
    return motor_elo

motor_elo_global = calcular_elo_global(df_games)

# --- FUNCIONES AUXILIARES ---
def obtener_clima_estadio(equipo_local):
    """Consulta el clima en vivo o aplica respaldo estacional según la región"""
    info = estadios_info.get(equipo_local, {"lat": 0, "lon": 0, "dome": False})
    
    if info["dome"]: 
        return 70.0, 0.0, True
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={info['lat']}&longitude={info['lon']}&current_weather=true"
        res = requests.get(url, timeout=3).json()
        if "current_weather" in res:
            temp_c = res["current_weather"]["temperature"]
            temp_f = (temp_c * 9/5) + 32
            wind_kmh = res["current_weather"]["windspeed"]
            wind_mph = wind_kmh * 0.621371
            return round(temp_f, 1), round(wind_mph, 1), False
    except:
        pass
    
    temp_base = 65.0
    viento_base = 6.0
    if equipo_local in ["BUF", "GB", "NE", "CHI", "MIN"]:
        temp_base = 42.0
        viento_base = 12.0
    elif equipo_local in ["MIA", "TB", "NO", "JAX", "ATL", "CAR"]:
        temp_base = 82.0
        viento_base = 5.0
        
    return temp_base, viento_base, False

def obtener_calendario_nfl(temporada, semana):
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

# --- PESTAÑAS DE NAVEGACIÓN ---
pestana_escanner, pestana_qbs, pestana_power = st.tabs(["🤖 Escáner de Jornada (EV+)", "🎯 Analizador de Yardas (QB Props)", "📈 Power Ranking ELO"])

# ==========================================
# 1. PESTAÑA: ESCÁNER DE JORNADA
# ==========================================
with pestana_escanner:
    st.markdown("### 🤖 Escáner Automático de Jornada (Filtro Estricto 60%+)")

    col_auto1, col_auto2, col_auto3 = st.columns(3)
    with col_auto1:
        temporada_auto = st.number_input("Temporada:", min_value=2020, max_value=2030, value=2026)
    with col_auto2:
        semana_auto = st.number_input("Semana a escanear:", min_value=1, max_value=22, value=1)
    with col_auto3:
        min_prob_filtro = st.slider("Filtro de Probabilidad Mínima (%):", min_value=50, max_value=80, value=60, step=1)

    if st.button("🚀 Escanear Toda la Semana (Multi-Motor)", type="primary"):
        with st.spinner("Analizando ELO, Montecarlo, Clima y Machine Learning..."):
            juegos_df = obtener_calendario_nfl(temporada_auto, semana_auto)
            
            if juegos_df.empty:
                st.warning(f"⚠️ No se encontraron partidos para la Temporada {temporada_auto}, Semana {semana_auto}.")
            else:
                ml_engine = PredictorNFL_ML()
                ml_engine.entrenar(df_games, df_qbs)
                
                apuestas_destacadas = []
                detalles_juegos = []
                
                for _, j in juegos_df.iterrows():
                    home_code = j.get("home_team")
                    away_code = j.get("away_team")
                    fecha_partido = str(j.get("gameday", "Fecha por confirmar"))
                    
                    if not home_code or not away_code:
                        continue
                        
                    temp, wind, is_dome = obtener_clima_estadio(home_code)
                    clima_str = "🏠 Domo (Controlado)" if is_dome else f"🌡️ {temp}°F | 💨 {wind} mph"
                    
                    linea_ou_api = float(j.get("total", 45.5)) if pd.notna(j.get("total")) else 45.5
                    spread_api = float(j.get("spread_line", -3.0)) if pd.notna(j.get("spread_line")) else -3.0
                    
                    mc = simular_nfl_montecarlo(home_code, away_code, df_games, linea_ou_api, spread_api)
                    ml = ml_engine.predecir_contexto(semana_auto, home_code, temp, wind, 1 if is_dome else 0)
                    
                    elo_h = motor_elo_global.ratings.get(home_code, 1500)
                    elo_a = motor_elo_global.ratings.get(away_code, 1500)
                    prob_elo_home = motor_elo_global.calcular_probabilidad_elo(elo_h, elo_a) * 100
                    
                    total_mc = mc['Proyeccion_Score']['Total_Proyectado']
                    total_ml = ml['ML_Puntos_Totales_Esperados']
                    prob_over = mc['Over_Under']['Prob Over']
                    prob_under = mc['Over_Under']['Prob Under']
                    
                    es_over = (total_mc > linea_ou_api) and (total_ml > linea_ou_api) and (prob_over >= min_prob_filtro)
                    es_under = (total_mc < linea_ou_api) and (total_ml < linea_ou_api) and (prob_under >= min_prob_filtro)
                    
                    veredicto = "Neutral"
                    if es_over:
                        veredicto = f"OVER {linea_ou_api} ({prob_over}% prob)"
                        apuestas_destacadas.append(f"🔥 **{away_code} @ {home_code}** ➔ Recomendación: **OVER {linea_ou_api}** ({prob_over}% de efectividad)")
                    elif es_under:
                        veredicto = f"UNDER {linea_ou_api} ({prob_under}% prob)"
                        apuestas_destacadas.append(f"🔥 **{away_code} @ {home_code}** ➔ Recomendación: **UNDER {linea_ou_api}** ({prob_under}% de efectividad)")
                        
                    detalles_juegos.append({
                        "juego": f"{away_code} @ {home_code}",
                        "fecha": fecha_partido,
                        "linea": linea_ou_api,
                        "mc": mc, "ml": ml, "elo_h": elo_h, "elo_a": elo_a, "prob_elo": prob_elo_home,
                        "temp": temp, "wind": wind, "is_dome": is_dome, "clima_str": clima_str,
                        "veredicto": veredicto, "es_recomendado": (es_over or es_under)
                    })

                # --- RESUMEN DIRECTO ---
                st.write("---")
                if apuestas_destacadas:
                    st.success(f"🎯 **¡Se encontraron {len(apuestas_destacadas)} Apuestas de Oro con Consenso Blindado!**")
                    for ap in apuestas_destacadas:
                        st.markdown(f"- {ap}")
                else:
                    st.warning("⚠️ No se encontraron partidos que cumplan estrictamente con el filtro del porcentaje mínimo para esta semana.")

                st.write("---")
                st.markdown("### 📋 Desglose Completo de la Jornada")
                
                for d in detalles_juegos:
                    with st.expander(f"📅 {d['fecha']} | 🏈 {d['juego']} | Estado: {d['veredicto']}"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown("**📊 Montecarlo & ELO**")
                            score_dict = d['mc']['Proyeccion_Score']
                            equipos_en_juego = [k for k in score_dict.keys() if k != 'Total_Proyectado']
                            eq_visita_j = equipos_en_juego[0] if len(equipos_en_juego) > 0 else 'Visita'
                            eq_local_j = equipos_en_juego[1] if len(equipos_en_juego) > 1 else 'Local'
                            
                            # Convertir a enteros redondeados
                            p_visita = int(round(score_dict.get(eq_visita_j, 0), 0))
                            p_local = int(round(score_dict.get(eq_local_j, 0), 0))
                            total_pts_entero = int(round(score_dict.get('Total_Proyectado', 0), 0))
                            
                            st.write(f"- Proyección Score: **{eq_visita_j} {p_visita} - {p_local} {eq_local_j}**")
                            st.write(f"- Total Proyectado: **{total_pts_entero} pts**")
                            st.write(f"- Prob. Victoria ELO: **{round(d['prob_elo'], 1)}%**")
                            st.write(f"- Prob. Over / Under: **{d['mc']['Over_Under']['Prob Over']}% / {d['mc']['Over_Under']['Prob Under']}%**")
                        with col_d2:
                            st.markdown("**🤖 Machine Learning & Clima**")
                            st.write(f"- Clima: {d['clima_str']}")
                            # Redondear puntos ajustados por ML a enteros
                            puntos_ml_entero = int(round(d['ml']['ML_Puntos_Totales_Esperados'], 0))
                            margen_ml_entero = round(d['ml']['ML_Margen_Local_Esperado'], 1)
                            
                            st.write(f"- Puntos Ajustados IA: **{puntos_ml_entero} pts**")
                            st.write(f"- Margen Local Previsto: **{margen_ml_entero} pts**")

# ==========================================
# 2. PESTAÑA: ANALIZADOR DE YARDAS DE QBS
# ==========================================
with pestana_qbs:
    st.markdown("### 🎯 Analizador de Yardas por Pase (Quarterback Props)")
    if df_qbs.empty:
        st.error("No hay registros históricos de QBs disponibles.")
    else:
        lista_qbs = sorted(list(df_qbs['player_name'].dropna().unique()))
        qb_seleccionado = st.selectbox("Selecciona o escribe el nombre del Quarterback:", lista_qbs)
        linea_yardas_lv = st.number_input("Línea de Yardas por Pase de Las Vegas:", min_value=100.0, max_value=400.0, value=245.5, step=0.5)
        
        if st.button("Simular Props de Yardas", type="primary"):
            qb_engine = PredictorYardasQB(df_qbs)
            resultado_qb = qb_engine.proyectar_yardas_qb(qb_seleccionado, linea_yardas_lv)
            if "error" in resultado_qb:
                st.error(resultado_qb["error"])
            else:
                st.success(f"📈 Análisis completado para **{resultado_qb['QB']}**")
                qb_col1, qb_col2, qb_col3 = st.columns(3)
                qb_col1.metric("Promedio Reciente", f"{resultado_qb['Yardas_Promedio_Recientes']} yds")
                qb_col2.metric("Prob. Over", f"{resultado_qb['Prob_Over_Yardas']}%")
                qb_col3.metric("Prob. Under", f"{resultado_qb['Prob_Under_Yardas']}%")

# ==========================================
# 3. PESTAÑA: POWER RANKING ELO
# ==========================================
with pestana_power:
    st.markdown("### 📈 Power Ranking ELO Actualizado de la NFL")
    st.write("Tabla general de poder de los 32 equipos basada en el historial de rendimiento y margen de victoria.")
    
    ranking = motor_elo_global.obtener_power_ranking()
    df_ranking = pd.DataFrame(ranking, columns=["Equipo", "Rating ELO"])
    df_ranking['Rating ELO'] = df_ranking['Rating ELO'].round(1)
    df_ranking.index = range(1, len(df_ranking) + 1)
    
    st.dataframe(df_ranking, use_container_width=True, height=600)
