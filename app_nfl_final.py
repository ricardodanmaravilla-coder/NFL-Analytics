import datetime as dt
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nfl_data_py as nfl

from modules.nfl_calibration import (
    empirical_residual_gt,
    empirical_residual_two_way,
    historico_antes,
    primary_with_agreement,
)
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_qb_engine import PredictorYardasQB
from modules.nfl_xgb_engine import PredictorXGBoostSpread

st.set_page_config(page_title="NFL Analytics Real V2", layout="wide", page_icon="🏈")
st.title("🏈 NFL Analytics Real V2")
st.caption("Datos reales + corte temporal estricto + calibración OOS. Dato faltante o contexto dudoso = NO BET.")

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

VALIDATION = {
    "moneyline_2025_picks": 39, "moneyline_2025_wins": 25, "moneyline_2025_roi": 4.03,
    "favorite_wf_picks": 87, "favorite_wf_wins": 63, "favorite_wf_roi": 13.07,
    "spread_ml_accuracy": 53.14, "xgb_accuracy": 51.13, "ou_2025_roi": -3.19,
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
    s = ia + ib
    return (ia / s, ib / s) if s > 0 else (None, None)


def two_way(a, b):
    if a is None or b is None:
        return None, None
    s = float(a) + float(b)
    return (100 * float(a) / s, 100 * float(b) / s) if s > 0 else (None, None)


def value_metrics(pct, odd, mkt):
    d = american_to_decimal(odd)
    if d is None or mkt is None or pct is None:
        return None
    return {"ev": ((pct / 100) * d - 1) * 100, "edge": (pct / 100 - mkt) * 100, "decimal": d}


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
    is_favorite = odd < 0
    return {
        "Acción": "BET" if is_favorite else "LEAN — NO AUTO BET",
        "Partido": partido, "Mercado": "Moneyline", "Apuesta": apuesta,
        "Prob calibrada": round(p, 1),
        "Soporte Elo/Emp": f"{support_probs[0]:.1f}% / {support_probs[1]:.1f}%",
        "Desacuerdo pp": round(disagreement, 1),
        "Momio real": int(odd), "Edge no-vig pp": round(vm["edge"], 2),
        "EV %": round(vm["ev"], 2),
        "_favorite": is_favorite,
        "_score": 1.5 * vm["edge"] + vm["ev"] - 0.3 * disagreement,
    }


@st.cache_data(ttl=3600)
def cargar_historico():
    games = pd.read_csv("data/historico_nfl_games.csv")
    qbs = pd.read_csv("data/historico_nfl_qbs.csv")
    pbp_path = "data/historico_nfl_pbp_team_game.csv"
    pbp = pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    return games, qbs, pbp


@st.cache_data(ttl=300)
def cargar_schedule(season):
    try:
        return nfl.import_schedules([int(season)])
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def motores(df, pbp):
    ml = PredictorNFL_ML()
    ml_ok = ml.entrenar(df, df_pbp_team_game=pbp)
    xgb = PredictorXGBoostSpread()
    xgb_ok = xgb.entrenar(df)
    elo = MotorELONFL()
    elo.actualizar_ratings(df)
    return ml, ml_ok, xgb, xgb_ok, elo


@st.cache_resource
def motor_elo_global(df):
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


@st.cache_data(ttl=1800)
def forecast_kickoff(team, gameday, gametime, roof=None):
    info = ESTADIOS.get(team)
    if not info:
        return None, None, False, "Sin coordenadas verificadas"
    lat, lon, default_dome = info
    dome = roof_is_dome(roof, default_dome)
    if dome:
        return None, None, True, f"Techo {roof or 'cerrado/domo'}"
    try:
        date = pd.to_datetime(gameday).date()
        hhmm = str(gametime or "13:00")[:5]
        et = dt.datetime.combine(date, dt.time.fromisoformat(hhmm), tzinfo=ZoneInfo("America/New_York"))
        utc = et.astimezone(ZoneInfo("UTC"))
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "UTC", "start_date": utc.date().isoformat(), "end_date": utc.date().isoformat(),
        }
        data = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8).json()
        times = pd.to_datetime(data.get("hourly", {}).get("time", []), utc=True)
        if not len(times):
            return None, None, False, "Forecast no disponible"
        idx = int(np.argmin(np.abs(times - pd.Timestamp(utc))))
        t = num(data["hourly"]["temperature_2m"][idx])
        w = num(data["hourly"]["wind_speed_10m"][idx])
        return t, w, False, f"{t:.1f}°F, {w:.1f} mph al kickoff"
    except Exception:
        return None, None, False, "Forecast no disponible"


df_games, df_qbs, df_pbp = cargar_historico()
elo_global = motor_elo_global(df_games)

with st.expander("✅ Datos y validación", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PBP real", "ACTIVO" if not df_pbp.empty else "NO DISPONIBLE", f"{len(df_pbp):,} equipo-partidos")
    c2.metric("ML favoritos WF", f"{VALIDATION['favorite_wf_wins']}/{VALIDATION['favorite_wf_picks']}", f"ROI agregado +{VALIDATION['favorite_wf_roi']}%")
    c3.metric("Spread", "NO AUTO BET", f"ML {VALIDATION['spread_ml_accuracy']}% / XGB {VALIDATION['xgb_accuracy']}%")
    c4.metric("O/U", "NO AUTO BET", f"ROI previo {VALIDATION['ou_2025_roi']}%")
    st.caption("Arranque ligero: ML/XGBoost sólo se entrenan al pulsar Analizar jornada. Filtro Moneyline: favoritos que pasan probabilidad + acuerdo + no-vig + EV se marcan BET; underdogs quedan LEAN / NO AUTO BET.")

scan_tab, qb_tab, elo_tab = st.tabs(["🤖 Scanner", "🎯 QB Props", "📈 ELO"])

with scan_tab:
    a, b, c = st.columns(3)
    season = a.number_input("Temporada", 2021, 2030, 2026, 1)
    week = b.number_input("Semana", 1, 22, 1, 1)
    top_n = c.slider("Máximo de BET Moneyline", 1, 10, 3)

    if st.button("Analizar jornada", type="primary"):
        sched = cargar_schedule(season)
        if sched.empty:
            st.error("No se pudo obtener el schedule real.")
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
            with st.spinner("Entrenando modelos con datos anteriores a esta jornada..."):
                ml_engine, ml_ok, xgb_engine, xgb_ok, elo_engine = motores(past_games, past_pbp)
            picks, diag = [], []
            if not ml_ok:
                st.warning("No hay histórico prepartido suficiente para entrenar el modelo en este corte temporal.")

            for _, g in games.iterrows():
                home, away = g.get("home_team"), g.get("away_team")
                if not home or not away:
                    continue
                partido = f"{away} @ {home}"
                hm, am = num(g.get("home_moneyline")), num(g.get("away_moneyline"))
                line, spread = num(g.get("total_line")), num(g.get("spread_line"))
                hr, ar = num(g.get("home_rest")), num(g.get("away_rest"))
                roof = g.get("roof")
                neutral = str(g.get("location", "")).strip().lower() == "neutral"
                if neutral:
                    diag.append({"Partido": partido, "Estado": "NO BET — sede neutral requiere estadio/contexto verificado", "Spread": spread, "Total": line})
                    continue

                temp, wind, dome, wmsg = forecast_kickoff(home, g.get("gameday"), g.get("gametime"), roof)
                ml = ml_engine.predecir_contexto(week, home, away, temp, wind, dome, hr, ar) if ml_ok else None
                emp = simular_nfl_montecarlo(home, away, past_games, line, spread)
                row = {
                    "Partido": partido, "PBP real": bool(ml and ml.get("Usa_PBP_Real")),
                    "Clima kickoff": wmsg, "ML odds H/A": f"{hm}/{am}", "Spread": spread, "Total": line,
                    "Histórico visible": len(past_games),
                }
                if not ml or not emp.get("Disponible"):
                    row["Estado"] = "Sin datos/modelos suficientes"
                    diag.append(row)
                    continue

                pml_h, pml_a = empirical_residual_two_way(
                    ml.get("ML_Margen_Local_Esperado"), 0.0, ml_engine.residuales_margen
                )
                pe_h, pe_a = two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
                pelo = 100 * elo_engine.calcular_probabilidad_elo(
                    elo_engine.ratings.get(home, 1500), elo_engine.ratings.get(away, 1500)
                )

                if hm is not None and am is not None and pml_h is not None and pml_a is not None and pe_h is not None and pe_a is not None:
                    ch = moneyline_candidate(partido, f"{home} ML", pml_h, [pelo, pe_h], hm, am)
                    ca = moneyline_candidate(partido, f"{away} ML", pml_a, [100 - pelo, pe_a], am, hm)
                    if ch:
                        picks.append(ch)
                    if ca:
                        picks.append(ca)

                if spread is not None:
                    ml_cov = empirical_residual_gt(ml.get("ML_Margen_Local_Esperado"), spread, ml_engine.residuales_margen)
                    e_h, _ = two_way(emp["Spread"].get("Cubre Local"), emp["Spread"].get("Cubre Visita"))
                    xh = xgb_engine.predecir_probabilidad_cover(week, spread, line, home, away, hr, ar) if xgb_ok and line is not None else None
                    row["Spread probs Cal/Emp/XGB"] = f"{None if ml_cov is None else round(ml_cov,1)} / {None if e_h is None else round(e_h,1)} / {None if xh is None else round(xh,1)}"
                    row["Spread estado"] = "NO AUTO BET — requiere edge walk-forward"

                weather_ok = dome or (temp is not None and wind is not None)
                if line is not None and weather_ok:
                    mo = empirical_residual_gt(ml.get("ML_Puntos_Totales_Esperados"), line, ml_engine.residuales_total)
                    eo, _ = two_way(emp["Over_Under"].get("Prob Over"), emp["Over_Under"].get("Prob Under"))
                    row["O/U probs Cal/Emp"] = f"{None if mo is None else round(mo,1)} / {None if eo is None else round(eo,1)}"
                    row["O/U estado"] = "NO AUTO BET — requiere edge walk-forward"

                row["ML prob calibrada H/A"] = f"{pml_h}/{pml_a}"
                row["Elo H"] = round(pelo, 1)
                row["Empírico H"] = None if pe_h is None else round(pe_h, 1)
                row["Estado"] = "Analizado sin look-ahead"
                diag.append(row)

            if picks:
                all_candidates = pd.DataFrame(picks).sort_values("_score", ascending=False)
                bets = all_candidates[all_candidates["_favorite"]].head(top_n).drop(columns=["_favorite", "_score"])
                leans = all_candidates[~all_candidates["_favorite"]].drop(columns=["_favorite", "_score"])
                if not bets.empty:
                    st.success(f"{len(bets)} BET Moneyline pasan filtros OOS y filtro estable de favoritos.")
                    st.dataframe(bets, width="stretch", hide_index=True)
                else:
                    st.info("No hay favoritos Moneyline con valor suficientemente robusto para BET en este corte temporal.")
                if not leans.empty:
                    st.markdown("### LEAN — underdogs con señal, no auto bet")
                    st.caption("Se muestran para diagnóstico; el walk-forward 2023–2025 no respaldó underdogs como filtro automático estable.")
                    st.dataframe(leans, width="stretch", hide_index=True)
            else:
                st.info("No hay Moneyline con valor suficientemente robusto en este corte temporal.")
            if diag:
                st.markdown("### Diagnóstico completo")
                st.dataframe(pd.DataFrame(diag), width="stretch", hide_index=True)

with qb_tab:
    st.info("QB props permanecen manuales y descriptivos; no se presentan como probabilidad calibrada de apuesta.")
    if not df_qbs.empty:
        qb = st.selectbox("Quarterback", sorted(df_qbs["player_name"].dropna().astype(str).unique()))
        line = st.number_input("Línea REAL passing yards", 100.0, 450.0, 245.5, 0.5)
        x, y = st.columns(2)
        over = x.number_input("Momio real Over (0=sin dato)", value=0, step=5)
        under = y.number_input("Momio real Under (0=sin dato)", value=0, step=5)
        if st.button("Analizar prop"):
            res = PredictorYardasQB(df_qbs).proyectar_yardas_qb(qb, line)
            st.json(res)
            if "error" not in res and over != 0 and under != 0:
                po, pu = no_vig(over, under)
                vo = value_metrics(res["Prob_Over_Yardas"], over, po)
                vu = value_metrics(res["Prob_Under_Yardas"], under, pu)
                st.write({"Frecuencia Over edge pp (no calibrada)": round(vo["edge"], 2), "Frecuencia Under edge pp (no calibrada)": round(vu["edge"], 2)})

with elo_tab:
    rank = pd.DataFrame(elo_global.obtener_power_ranking(), columns=["Equipo", "ELO"])
    rank["ELO"] = rank["ELO"].round(1)
    st.dataframe(rank, width="stretch", hide_index=True)
