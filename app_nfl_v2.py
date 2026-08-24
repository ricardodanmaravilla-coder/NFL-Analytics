import datetime as dt
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl

from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_qb_engine import PredictorYardasQB
from modules.nfl_xgb_engine import PredictorXGBoostSpread

st.set_page_config(page_title="NFL Analytics V2", layout="wide", page_icon="🏈")
st.title("🏈 NFL Analytics V2 — datos reales, sin cuotas inventadas")
st.caption("Las recomendaciones solo se generan cuando existen línea y precio reales. Datos faltantes = NO BET.")

ESTADIOS = {
    "BUF": (42.773, -78.786, False), "MIA": (25.957, -80.238, False), "NE": (42.090, -71.264, False),
    "NYJ": (40.813, -74.074, False), "NYG": (40.813, -74.074, False), "BAL": (39.277, -76.622, False),
    "CIN": (39.095, -84.516, False), "CLE": (41.506, -81.699, False), "PIT": (40.446, -80.015, False),
    "HOU": (29.684, -95.410, True), "IND": (39.760, -86.163, True), "JAX": (30.323, -81.637, False),
    "TEN": (36.166, -86.771, False), "DEN": (39.743, -105.020, False), "KC": (39.048, -94.483, False),
    "LV": (36.090, -115.183, True), "LAC": (33.953, -118.339, True), "DAL": (32.747, -97.092, True),
    "PHI": (39.900, -75.167, False), "WAS": (38.907, -76.864, False), "CHI": (41.862, -87.616, False),
    "DET": (42.340, -83.045, True), "GB": (44.501, -88.062, False), "MIN": (44.973, -93.257, True),
    "ATL": (33.755, -84.400, True), "CAR": (35.225, -80.852, False), "NO": (29.951, -90.081, True),
    "TB": (27.975, -82.503, False), "ARI": (33.527, -112.262, True), "LA": (33.953, -118.339, True),
    "LAR": (33.953, -118.339, True), "SF": (37.403, -121.969, False), "SEA": (47.595, -122.331, False),
}


def _num(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def american_to_decimal(v):
    v = _num(v)
    if v is None or v == 0:
        return None
    return 1.0 + (v / 100.0 if v > 0 else 100.0 / abs(v))


def no_vig(odds_a, odds_b):
    da, db = american_to_decimal(odds_a), american_to_decimal(odds_b)
    if da is None or db is None:
        return None, None
    ia, ib = 1 / da, 1 / db
    s = ia + ib
    return ia / s, ib / s


def normal_gt(mean, threshold, sigma):
    if mean is None or threshold is None or sigma is None or sigma <= 0:
        return None
    z = (float(mean) - float(threshold)) / float(sigma)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0


def conditional_two_way(a, b):
    if a is None or b is None:
        return None, None
    s = float(a) + float(b)
    if s <= 0:
        return None, None
    return float(a) / s * 100.0, float(b) / s * 100.0


def evaluar_valor(prob_pct, american, market_prob, min_prob=54.0, min_edge=3.0, min_ev=3.0):
    dec = american_to_decimal(american)
    if dec is None or market_prob is None or prob_pct is None:
        return None
    ev = (prob_pct / 100.0) * dec - 1.0
    edge = prob_pct / 100.0 - market_prob
    return {
        "ok": prob_pct >= min_prob and edge >= min_edge / 100.0 and ev >= min_ev / 100.0,
        "decimal": dec,
        "ev_pct": ev * 100.0,
        "edge_pp": edge * 100.0,
    }


@st.cache_data(ttl=3600)
def cargar_historico():
    games = pd.read_csv("data/historico_nfl_games.csv")
    qbs = pd.read_csv("data/historico_nfl_qbs.csv")
    return games, qbs


@st.cache_data(ttl=300)
def cargar_schedule(season):
    try:
        return nfl.import_schedules([int(season)])
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def entrenar_motores(df_games):
    ml = PredictorNFL_ML()
    ml_ok = ml.entrenar(df_games)
    xgb = PredictorXGBoostSpread()
    xgb_ok = xgb.entrenar(df_games)
    elo = MotorELONFL()
    elo.actualizar_ratings(df_games)
    return ml, ml_ok, xgb, xgb_ok, elo


@st.cache_data(ttl=1800)
def forecast_kickoff(team, gameday, gametime):
    info = ESTADIOS.get(team)
    if not info:
        return None, None, False, "Estadio sin coordenadas"
    lat, lon, dome = info
    if dome:
        return None, None, True, "Domo / clima exterior no aplicado"
    try:
        date_obj = pd.to_datetime(gameday).date()
        hhmm = str(gametime or "13:00")[:5]
        kickoff_et = dt.datetime.combine(date_obj, dt.time.fromisoformat(hhmm), tzinfo=ZoneInfo("America/New_York"))
        kickoff_utc = kickoff_et.astimezone(ZoneInfo("UTC"))
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "UTC",
            "start_date": kickoff_utc.date().isoformat(),
            "end_date": kickoff_utc.date().isoformat(),
        }
        data = requests.get(url, params=params, timeout=8).json()
        times = pd.to_datetime(data.get("hourly", {}).get("time", []), utc=True)
        if len(times) == 0:
            return None, None, False, "Forecast no disponible"
        idx = int(np.argmin(np.abs(times - pd.Timestamp(kickoff_utc))))
        temp = _num(data["hourly"]["temperature_2m"][idx])
        wind = _num(data["hourly"]["wind_speed_10m"][idx])
        return temp, wind, False, f"Forecast kickoff: {temp:.1f}°F, {wind:.1f} mph"
    except Exception:
        return None, None, False, "Forecast no disponible"


def agregar_candidato(lista, partido, mercado, apuesta, probs, odds_self, odds_other, min_prob=54.0, min_edge=3.0, min_ev=3.0):
    probs = [float(p) for p in probs if p is not None]
    if len(probs) < 2:
        return
    disagreement = max(probs) - min(probs)
    if disagreement > 15.0:
        return
    p = float(np.mean(probs))
    mkt_self, _ = no_vig(odds_self, odds_other)
    value = evaluar_valor(p, odds_self, mkt_self, min_prob=min_prob, min_edge=min_edge, min_ev=min_ev)
    if not value or not value["ok"]:
        return
    lista.append({
        "Partido": partido,
        "Mercado": mercado,
        "Apuesta": apuesta,
        "Prob. consenso": round(p, 1),
        "Rango modelos": f"{min(probs):.1f}-{max(probs):.1f}%",
        "Desacuerdo pp": round(disagreement, 1),
        "Momio real": int(float(odds_self)),
        "Edge no-vig pp": round(value["edge_pp"], 2),
        "EV %": round(value["ev_pct"], 2),
        "_score": value["edge_pp"] * 1.5 + value["ev_pct"] - disagreement * 0.2,
    })


df_games, df_qbs = cargar_historico()
ml_engine, ml_ok, xgb_engine, xgb_ok, elo_engine = entrenar_motores(df_games)

scanner_tab, qb_tab, elo_tab = st.tabs(["🤖 Scanner real", "🎯 QB Props manual", "📈 ELO"])

with scanner_tab:
    c1, c2, c3 = st.columns(3)
    season = c1.number_input("Temporada", min_value=2021, max_value=2030, value=2026, step=1)
    week = c2.number_input("Semana", min_value=1, max_value=22, value=1, step=1)
    top_n = c3.slider("Máximo de recomendaciones", 1, 10, 3)

    st.write(f"ML entrenado: {'✅' if ml_ok else '❌'} | XGBoost spread: {'✅' if xgb_ok else '⚠️ sin datos de mercado históricos suficientes'}")

    if st.button("Analizar jornada con datos reales", type="primary"):
        schedule = cargar_schedule(season)
        if schedule.empty:
            st.error("No se pudo obtener el schedule real de nflverse.")
        else:
            games = schedule[(schedule["week"] == int(week))].copy()
            if "game_type" in games.columns:
                games = games[games["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
            if "home_score" in games.columns:
                upcoming = games[games["home_score"].isna()].copy()
                if not upcoming.empty:
                    games = upcoming

            candidatos, diagnosticos = [], []
            for _, g in games.iterrows():
                home, away = g.get("home_team"), g.get("away_team")
                if not home or not away:
                    continue
                partido = f"{away} @ {home}"
                total_line = _num(g.get("total_line"))
                spread_line = _num(g.get("spread_line"))
                home_ml, away_ml = _num(g.get("home_moneyline")), _num(g.get("away_moneyline"))
                home_sp_odds, away_sp_odds = _num(g.get("home_spread_odds")), _num(g.get("away_spread_odds"))
                over_odds, under_odds = _num(g.get("over_odds")), _num(g.get("under_odds"))
                home_rest, away_rest = _num(g.get("home_rest")), _num(g.get("away_rest"))

                temp, wind, dome, weather_msg = forecast_kickoff(home, g.get("gameday"), g.get("gametime"))
                ml = ml_engine.predecir_contexto(week, home, away, temp, wind, dome, home_rest, away_rest) if ml_ok else None
                emp = simular_nfl_montecarlo(home, away, df_games, total_line, spread_line)

                diag = {
                    "Partido": partido, "Spread line nflverse": spread_line, "Total line": total_line,
                    "ML real H/A": f"{home_ml}/{away_ml}", "Clima": weather_msg,
                    "Empírico": bool(emp.get("Disponible")), "ML": bool(ml), "XGB": bool(xgb_ok),
                }

                if not emp.get("Disponible") or not ml:
                    diag["Estado"] = "Sin modelos suficientes"
                    diagnosticos.append(diag)
                    continue

                sigma_margin = ml.get("Sigma_Margen_OOS")
                sigma_total = ml.get("Sigma_Total_OOS")
                ml_home_win = normal_gt(ml.get("ML_Margen_Local_Esperado"), 0.0, sigma_margin)
                emp_home, emp_away = conditional_two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
                elo_home = elo_engine.calcular_probabilidad_elo(elo_engine.ratings.get(home, 1500), elo_engine.ratings.get(away, 1500)) * 100.0

                if home_ml is not None and away_ml is not None and ml_home_win is not None:
                    agregar_candidato(candidatos, partido, "Moneyline", f"{home} ML", [elo_home, ml_home_win, emp_home], home_ml, away_ml)
                    agregar_candidato(candidatos, partido, "Moneyline", f"{away} ML", [100-elo_home, 100-ml_home_win, emp_away], away_ml, home_ml)

                if spread_line is not None and home_sp_odds is not None and away_sp_odds is not None and sigma_margin:
                    ml_cover_home = normal_gt(ml.get("ML_Margen_Local_Esperado"), spread_line, sigma_margin)
                    emp_cover_home, emp_cover_away = conditional_two_way(emp["Spread"].get("Cubre Local"), emp["Spread"].get("Cubre Visita"))
                    xgb_home = xgb_engine.predecir_probabilidad_cover(week, spread_line, total_line, home, away, home_rest, away_rest) if xgb_ok and total_line is not None else None
                    if xgb_home is not None:
                        home_handicap = -spread_line
                        away_handicap = spread_line
                        agregar_candidato(candidatos, partido, "Spread", f"{home} {home_handicap:+.1f}", [ml_cover_home, emp_cover_home, xgb_home], home_sp_odds, away_sp_odds)
                        agregar_candidato(candidatos, partido, "Spread", f"{away} {away_handicap:+.1f}", [100-ml_cover_home, emp_cover_away, 100-xgb_home], away_sp_odds, home_sp_odds)

                # Para outdoor, no hay pick de total sin forecast real de kickoff.
                weather_ok = dome or (temp is not None and wind is not None)
                if total_line is not None and over_odds is not None and under_odds is not None and sigma_total and weather_ok:
                    ml_over = normal_gt(ml.get("ML_Puntos_Totales_Esperados"), total_line, sigma_total)
                    emp_over, emp_under = conditional_two_way(emp["Over_Under"].get("Prob Over"), emp["Over_Under"].get("Prob Under"))
                    agregar_candidato(candidatos, partido, "Total", f"Over {total_line}", [ml_over, emp_over], over_odds, under_odds, min_prob=53.5)
                    agregar_candidato(candidatos, partido, "Total", f"Under {total_line}", [100-ml_over, emp_under], under_odds, over_odds, min_prob=53.5)

                diag["Estado"] = "Analizado"
                diag["ML total"] = round(ml.get("ML_Puntos_Totales_Esperados"), 1)
                diag["Emp total"] = emp["Proyeccion_Score"].get("Total_Proyectado")
                diagnosticos.append(diag)

            if candidatos:
                out = pd.DataFrame(candidatos).sort_values("_score", ascending=False).head(top_n).drop(columns=["_score"])
                st.success(f"{len(out)} oportunidades superan consenso + no-vig + EV.")
                st.dataframe(out, use_container_width=True, hide_index=True)
            else:
                st.info("No hay apuestas que superen los filtros con datos reales disponibles.")

            if diagnosticos:
                st.markdown("### Auditoría de datos por partido")
                st.dataframe(pd.DataFrame(diagnosticos), use_container_width=True, hide_index=True)

with qb_tab:
    st.markdown("### QB Props — solo con línea y precios introducidos desde tu sportsbook")
    if df_qbs.empty:
        st.error("No hay histórico de QBs.")
    else:
        qb = st.selectbox("Quarterback", sorted(df_qbs["player_name"].dropna().astype(str).unique()))
        line = st.number_input("Línea real de passing yards", 100.0, 450.0, 245.5, 0.5)
        co1, co2 = st.columns(2)
        over_am = co1.number_input("Momio real Over (americano; 0 = no disponible)", value=0, step=5)
        under_am = co2.number_input("Momio real Under (americano; 0 = no disponible)", value=0, step=5)
        if st.button("Analizar prop real"):
            res = PredictorYardasQB(df_qbs).proyectar_yardas_qb(qb, line)
            if "error" in res:
                st.error(res["error"])
            else:
                st.json(res)
                if over_am != 0 and under_am != 0:
                    mkt_o, mkt_u = no_vig(over_am, under_am)
                    vo = evaluar_valor(res["Prob_Over_Yardas"], over_am, mkt_o, min_prob=0, min_edge=0, min_ev=0)
                    vu = evaluar_valor(res["Prob_Under_Yardas"], under_am, mkt_u, min_prob=0, min_edge=0, min_ev=0)
                    st.write({"Over edge_pp": round(vo["edge_pp"],2), "Over EV%": round(vo["ev_pct"],2), "Under edge_pp": round(vu["edge_pp"],2), "Under EV%": round(vu["ev_pct"],2)})

with elo_tab:
    ranking = pd.DataFrame(elo_engine.obtener_power_ranking(), columns=["Equipo", "ELO"])
    ranking["ELO"] = ranking["ELO"].round(1)
    st.dataframe(ranking, use_container_width=True, hide_index=True)
