import datetime as dt
import os
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl

from modules.nfl_calibration import empirical_residual_two_way, historico_antes, primary_with_agreement
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo

st.set_page_config(page_title="NFL Analytics Real V2", layout="wide", page_icon="🏈")
st.title("🏈 NFL Analytics Real V2")
st.caption("Scanner estable: sólo calcula al pulsar Analizar jornada. El runtime de producción entrena únicamente Moneyline/margen para evitar picos de memoria.")

ESTADIOS = {
    "BUF": (42.773,-78.786,False), "MIA": (25.957,-80.238,False), "NE": (42.090,-71.264,False),
    "NYJ": (40.813,-74.074,False), "NYG": (40.813,-74.074,False), "BAL": (39.277,-76.622,False),
    "CIN": (39.095,-84.516,False), "CLE": (41.506,-81.699,False), "PIT": (40.446,-80.015,False),
    "HOU": (29.684,-95.410,True), "IND": (39.760,-86.163,True), "JAX": (30.323,-81.637,False),
    "TEN": (36.166,-86.771,False), "DEN": (39.743,-105.020,False), "KC": (39.048,-94.483,False),
    "LV": (36.090,-115.183,True), "LAC": (33.953,-118.339,True), "DAL": (32.747,-97.092,True),
    "PHI": (39.900,-75.167,False), "WAS": (38.907,-76.864,False), "CHI": (41.862,-87.616,False),
    "DET": (42.340,-83.045,True), "GB": (44.501,-88.062,False), "MIN": (44.973,-93.257,True),
    "ATL": (33.755,-84.400,True), "CAR": (35.225,-80.852,False), "NO": (29.951,-90.081,True),
    "TB": (27.975,-82.503,False), "ARI": (33.527,-112.262,True), "LA": (33.953,-118.339,True),
    "LAR": (33.953,-118.339,True), "SF": (37.403,-121.969,False), "SEA": (47.595,-122.331,False),
}


def num(v):
    try:
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def american_to_decimal(v):
    x = num(v)
    if x is None or x == 0:
        return None
    return 1 + (x / 100 if x > 0 else 100 / abs(x))


def no_vig(a, b):
    da, db = american_to_decimal(a), american_to_decimal(b)
    if da is None or db is None:
        return None, None
    ia, ib = 1 / da, 1 / db
    total = ia + ib
    return (ia / total, ib / total) if total > 0 else (None, None)


def value_metrics(pct, odd, mkt):
    dec = american_to_decimal(odd)
    if dec is None or mkt is None or pct is None:
        return None
    return {"ev": ((pct / 100) * dec - 1) * 100, "edge": (pct / 100 - mkt) * 100}


def two_way(a, b):
    if a is None or b is None:
        return None, None
    total = float(a) + float(b)
    return (100 * float(a) / total, 100 * float(b) / total) if total > 0 else (None, None)


def moneyline_candidate(partido, apuesta, primary_prob, support_probs, odd_self, odd_other):
    p = primary_with_agreement(primary_prob, support_probs, max_disagreement=15.0)
    if p is None:
        return None
    probs = [float(p)] + [float(x) for x in support_probs if x is not None]
    disagreement = max(probs) - min(probs)
    mkt, _ = no_vig(odd_self, odd_other)
    vm = value_metrics(p, odd_self, mkt)
    odd = num(odd_self)
    if vm is None or odd is None or p < 54.0 or vm["edge"] < 3.0 or vm["ev"] < 3.0:
        return None
    return {
        "Partido": partido,
        "Apuesta": apuesta,
        "Probabilidad": round(p, 1),
        "Momio": int(odd),
        "Edge": round(vm["edge"], 2),
        "EV": round(vm["ev"], 2),
        "Desacuerdo": round(disagreement, 1),
        "_favorite": odd < 0,
        "_score": 1.5 * vm["edge"] + vm["ev"] - 0.3 * disagreement,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_historico():
    games = pd.read_csv("data/historico_nfl_games.csv")
    pbp_path = "data/historico_nfl_pbp_team_game.csv"
    pbp = pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    return games, pbp


@st.cache_data(ttl=300, show_spinner=False)
def cargar_schedule(season):
    try:
        return nfl.import_schedules([int(season)])
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def motor_ml(df, pbp):
    ml = MoneylineRuntime()
    return ml, ml.entrenar(df, df_pbp_team_game=pbp)


@st.cache_resource(show_spinner=False)
def motor_elo(df):
    elo = MotorELONFL()
    elo.actualizar_ratings(df)
    return elo


def roof_is_dome(roof, fallback=False):
    text = str(roof or "").strip().lower()
    if text in {"dome", "closed", "indoors", "indoor"}:
        return True
    if text in {"outdoors", "outdoor", "open"}:
        return False
    return bool(fallback)


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_kickoff(team, gameday, gametime, roof=None):
    info = ESTADIOS.get(team)
    if not info:
        return None, None, False, "Sin coordenadas verificadas"
    lat, lon, default_dome = info
    dome = roof_is_dome(roof, default_dome)
    if dome:
        return None, None, True, "Domo/techo cerrado"
    try:
        date = pd.to_datetime(gameday).date()
        hhmm = str(gametime or "13:00")[:5]
        et = dt.datetime.combine(date, dt.time.fromisoformat(hhmm), tzinfo=ZoneInfo("America/New_York"))
        utc = et.astimezone(ZoneInfo("UTC"))
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
            "start_date": utc.date().isoformat(), "end_date": utc.date().isoformat(),
        }
        data = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=6).json()
        times = pd.to_datetime(data.get("hourly", {}).get("time", []), utc=True)
        if not len(times):
            return None, None, False, "Forecast no disponible"
        idx = int(np.argmin(np.abs(times - pd.Timestamp(utc))))
        t = num(data["hourly"]["temperature_2m"][idx])
        w = num(data["hourly"]["wind_speed_10m"][idx])
        return t, w, False, f"{t:.1f}°F · {w:.1f} mph"
    except Exception:
        return None, None, False, "Forecast no disponible"


def guardar(bets, leans, diag, label):
    st.session_state["nfl_bets"] = bets
    st.session_state["nfl_leans"] = leans
    st.session_state["nfl_diag"] = diag
    st.session_state["nfl_label"] = label


def render():
    bets = st.session_state.get("nfl_bets")
    leans = st.session_state.get("nfl_leans")
    diag = st.session_state.get("nfl_diag")
    label = st.session_state.get("nfl_label")
    if bets is None and leans is None and diag is None:
        return
    st.divider()
    if label:
        st.caption(label)
    if isinstance(bets, pd.DataFrame) and not bets.empty:
        top = bets.iloc[0]
        st.subheader("⭐ Recomendación principal")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pick", top["Apuesta"])
        c2.metric("Probabilidad", f"{top['Probabilidad']:.1f}%")
        c3.metric("Edge", f"{top['Edge']:.2f} pp")
        c4.metric("EV", f"{top['EV']:.2f}%")
        st.caption(f"{top['Partido']} · momio {int(top['Momio'])}")
        if len(bets) > 1:
            st.markdown("#### Otras recomendaciones")
            st.dataframe(bets.iloc[1:].reset_index(drop=True), width="stretch", hide_index=True)
    else:
        st.info("No hay una recomendación suficientemente robusta en este corte temporal.")
    if isinstance(leans, pd.DataFrame) and not leans.empty:
        with st.expander("Señales secundarias — no auto bet"):
            st.dataframe(leans.reset_index(drop=True), width="stretch", hide_index=True)
    if isinstance(diag, pd.DataFrame) and not diag.empty:
        with st.expander("Diagnóstico completo"):
            st.dataframe(diag, width="stretch", hide_index=True)


df_games, df_pbp = cargar_historico()

with st.form("scanner_form", clear_on_submit=False):
    a, b = st.columns(2)
    season = a.number_input("Temporada", min_value=2021, max_value=2030, value=2026, step=1)
    week = b.number_input("Semana", min_value=1, max_value=22, value=1, step=1)
    submitted = st.form_submit_button("🔎 Analizar jornada", type="primary", use_container_width=True)

if submitted:
    label = f"Temporada {int(season)} · Semana {int(week)}"
    try:
        sched = cargar_schedule(season)
        if sched.empty:
            guardar(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), label)
            st.error("No se pudo obtener el calendario real.")
        else:
            games = sched[sched["week"] == int(week)].copy()
            if "game_type" in games.columns:
                games = games[games["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
            if "home_score" in games.columns:
                future = games[games["home_score"].isna()]
                if not future.empty:
                    games = future

            past_games = historico_antes(df_games, season, week)
            past_pbp = historico_antes(df_pbp, season, week) if not df_pbp.empty else pd.DataFrame()

            with st.spinner("Entrenando Moneyline y analizando la jornada..."):
                ml_engine, ml_ok = motor_ml(past_games, past_pbp)
                elo_engine = motor_elo(past_games)
                if not ml_ok:
                    raise RuntimeError("No hubo histórico suficiente para entrenar Moneyline en este corte temporal.")

                picks, diag = [], []
                for _, g in games.iterrows():
                    home, away = g.get("home_team"), g.get("away_team")
                    if not home or not away:
                        continue
                    partido = f"{away} @ {home}"
                    try:
                        hm, am = num(g.get("home_moneyline")), num(g.get("away_moneyline"))
                        if hm is None or am is None:
                            diag.append({"Partido": partido, "Estado": "Sin momios Moneyline"})
                            continue
                        if str(g.get("location", "")).strip().lower() == "neutral":
                            diag.append({"Partido": partido, "Estado": "NO BET — sede neutral"})
                            continue
                        hr, ar = num(g.get("home_rest")), num(g.get("away_rest"))
                        temp, wind, dome, wmsg = forecast_kickoff(home, g.get("gameday"), g.get("gametime"), g.get("roof"))
                        ml = ml_engine.predecir_contexto(week, home, away, temp, wind, dome, hr, ar)
                        emp = simular_nfl_montecarlo(home, away, past_games, num(g.get("total_line")), num(g.get("spread_line")))
                        if not ml or not emp.get("Disponible"):
                            diag.append({"Partido": partido, "Estado": "Sin datos suficientes", "Clima": wmsg})
                            continue
                        p_h, p_a = empirical_residual_two_way(ml.get("ML_Margen_Local_Esperado"), 0.0, ml_engine.residuales_margen)
                        e_h, e_a = two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
                        elo_h = 100 * elo_engine.calcular_probabilidad_elo(elo_engine.ratings.get(home, 1500), elo_engine.ratings.get(away, 1500))
                        ch = moneyline_candidate(partido, f"{home} ML", p_h, [elo_h, e_h], hm, am)
                        ca = moneyline_candidate(partido, f"{away} ML", p_a, [100 - elo_h, e_a], am, hm)
                        if ch:
                            picks.append(ch)
                        if ca:
                            picks.append(ca)
                        diag.append({"Partido": partido, "Prob H/A": f"{p_h}/{p_a}", "Clima": wmsg, "Estado": "Analizado"})
                    except Exception as game_error:
                        diag.append({"Partido": partido, "Estado": f"Error aislado: {type(game_error).__name__}"})

            if picks:
                candidates = pd.DataFrame(picks).sort_values("_score", ascending=False)
                bets = candidates[candidates["_favorite"]].drop(columns=["_favorite", "_score"])
                leans = candidates[~candidates["_favorite"]].drop(columns=["_favorite", "_score"])
            else:
                bets, leans = pd.DataFrame(), pd.DataFrame()
            guardar(bets, leans, pd.DataFrame(diag), label)
    except MemoryError:
        guardar(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), label)
        st.error("El servidor agotó memoria durante el análisis. El proceso fue detenido de forma segura.")
    except Exception as exc:
        guardar(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), label)
        st.error(f"El análisis no pudo completarse: {type(exc).__name__}: {exc}")

render()

with st.expander("Estado del sistema"):
    st.write({
        "PBP": "ACTIVO" if not df_pbp.empty else "NO DISPONIBLE",
        "Moneyline": "runtime ligero; BET automático sólo en favoritos validados",
        "Spread": "NO AUTO BET",
        "Over/Under": "NO AUTO BET",
    })
