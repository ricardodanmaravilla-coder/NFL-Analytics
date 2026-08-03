import streamlit as st
import pandas as pd
import os
import datetime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_ml_engine import PredictorNFL_ML

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="🏈 NFL Analytics & Value Betting (2026)", layout="wide")

st.title("🏈 NFL Analytics & Value Betting (2026)")
st.write("Simulador Híbrido: Montecarlo (Stats Base) + Machine Learning (Clima, Altitud, QB)")

# --- CARGA DE DATOS ---
@st.cache_data(ttl=3600)
def cargar_datos_nfl():
    url_games = 'https://raw.githubusercontent.com/ricardodanmaravilla-coder/NFL-Analytics/main/data/historico_nfl_games.csv'
    url_qbs = 'https://raw.githubusercontent.com/ricardodanmaravilla-coder/NFL-Analytics/main/data/historico_nfl_qbs.csv'
    
    df_games, df_qbs = pd.DataFrame(), pd.DataFrame()
    
    # Intentar carga local primero
    if os.path.exists('data/historico_nfl_games.csv'):
        df_games = pd.read_csv('data/historico_nfl_games.csv')
    else:
        try: df_games = pd.read_csv(url_games)
        except: pass
        
    if os.path.exists('data/historico_nfl_qbs.csv'):
        df_qbs = pd.read_csv('data/historico_nfl_qbs.csv')
    else:
        try: df_qbs = pd.read_csv(url_qbs)
        except: pass
        
    return df_games, df_qbs

df_games, df_qbs = cargar_datos_nfl()

if df_games.empty:
    st.error("🚨 No se pudo cargar la base de datos histórica. Ejecuta el Extractor primero.")
    st.stop()

# Diccionario estándar de equipos NFL para selección
equipos_nfl = sorted(list(df_games['home_team'].dropna().unique()))

# --- INTERFAZ DE USUARIO ---
st.markdown("### 1. Selecciona el Encuentro y Condiciones")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🏟️ Equipos")
    local = st.selectbox("Equipo Local (Home):", equipos_nfl, index=equipos_nfl.index('KC') if 'KC' in equipos_nfl else 0)
    visita = st.selectbox("Equipo Visitante (Away):", [eq for eq in equipos_nfl if eq != local], index=0)
    semana = st.number_input("Semana de la Temporada (Week):", min_value=1, max_value=22, value=1)

with col2:
    st.markdown("#### 🌪️ Contexto del Juego (Para Machine Learning)")
    is_dome = st.checkbox("¿Se juega en estadio techado (Dome)?", value=False)
    
    if is_dome:
        temp = 70.0
        wind = 0.0
        st.info("Al ser techado, el clima se controla a 70°F sin viento.")
    else:
        temp = st.slider("Temperatura esperada (°F):", min_value=-10.0, max_value=110.0, value=65.0)
        wind = st.slider("Viento esperado (mph):", min_value=0.0, max_value=50.0, value=5.0)

st.markdown("---")
st.markdown("### 2. Líneas de Las Vegas (Para evaluar el Valor/EV+)")
col_lv1, col_lv2 = st.columns(2)
with col_lv1:
    linea_ou = st.number_input("Línea Total Over/Under (Ej. 45.5):", min_value=30.0, max_value=60.0, value=45.5, step=0.5)
with col_lv2:
    # Spread del local: Ej. Si Kansas City es favorito por 3 puntos, pones -3.0
    spread_local = st.number_input("Spread del Local (Ej. -3.0 si es favorito, +3.0 si es underdog):", min_value=-25.0, max_value=25.0, value=-3.0, step=0.5)

if st.button("Ejecutar Simulador Híbrido NFL", type="primary"):
    with st.spinner("Simulando partido 10,000 veces y consultando a la IA..."):
        
        # 1. MOTOR MONTECARLO (Stats Base)
        resultados_mc = simular_nfl_montecarlo(
            local=local, 
            visita=visita, 
            df_games=df_games, 
            linea_ou=linea_ou, 
            spread_local=spread_local, 
            n_simulaciones=10000
        )
        
        # 2. MOTOR MACHINE LEARNING (Contexto y Clima)
        ml_engine = PredictorNFL_ML()
        modelo_listo = ml_engine.entrenar(df_games, df_qbs)
        
        resultados_ml = None
        if modelo_listo:
            resultados_ml = ml_engine.predecir_contexto(
                week=semana, 
                home_team=local, 
                temp=temp, 
                wind=wind, 
                is_dome=1 if is_dome else 0
            )

        # --- MOSTRAR RESULTADOS ---
        st.markdown("---")
        st.subheader(f"📊 Resultados de la Simulación: {visita} @ {local}")
        
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        mc_col1.metric("Probabilidad Gana Local (MC)", f"{resultados_mc['Moneyline']['Gana Local']}%")
        mc_col2.metric(f"Prob. {local} cubre {spread_local} (MC)", f"{resultados_mc['Spread']['Cubre Local']}%")
        mc_col3.metric(f"Prob. Over {linea_ou} (MC)", f"{resultados_mc['Over_Under']['Prob Over']}%")

        if resultados_ml:
            st.markdown("### 🤖 Validación de Contexto (Machine Learning)")
            st.info("El ML ajusta la expectativa de Montecarlo analizando cómo el Clima, la Altitud y la Semana de temporada afectan históricamente.")
            
            ml_col1, ml_col2 = st.columns(2)
            
            puntos_totales_mc = resultados_mc['Proyeccion_Score']['Total_Proyectado']
            puntos_totales_ml = resultados_ml['ML_Puntos_Totales_Esperados']
            
            # Margen Local: Si es positivo, gana local. Si es negativo, gana visita.
            margen_ml = resultados_ml['ML_Margen_Local_Esperado']
            favorito_ml = local if margen_ml > 0 else visita
            margen_ml_abs = abs(margen_ml)
            
            with ml_col1:
                st.markdown("**🎯 Totales (Over/Under)**")
                st.write(f"- Proyección Pura (Montecarlo): **{puntos_totales_mc} pts**")
                st.write(f"- Ajuste por Clima/Altitud (ML): **{puntos_totales_ml} pts**")
                
                # Consenso Totales
                if puntos_totales_mc > linea_ou and puntos_totales_ml > linea_ou:
                    st.success("✅ **CONSENSO BLINDADO:** Ambos motores proyectan OVER.")
                elif puntos_totales_mc < linea_ou and puntos_totales_ml < linea_ou:
                    st.success("✅ **CONSENSO BLINDADO:** Ambos motores proyectan UNDER.")
                else:
                    st.warning("⚠️ **ALERTA DE CONFLICTO:** Los motores no coinciden por el clima. Riesgo alto.")

            with ml_col2:
                st.markdown("**🏈 Hándicap (Spread)**")
                st.write(f"- ML predice victoria de: **{favorito_ml} por {margen_ml_abs} pts**")
                
                # Evaluación básica del spread
                cubre_ml_local = margen_ml > abs(spread_local) if spread_local < 0 else margen_ml > -spread_local
                if (resultados_mc['Spread']['Cubre Local'] > 55.0) and cubre_ml_local:
                    st.success(f"✅ **CONSENSO BLINDADO:** {local} cubre el spread de {spread_local}.")
                elif (resultados_mc['Spread']['Cubre Visita'] > 55.0) and not cubre_ml_local:
                    st.success(f"✅ **CONSENSO BLINDADO:** {visita} cubre su spread.")
                else:
                    st.warning("⚠️ **Sin valor claro en el Spread.** Paso.")
