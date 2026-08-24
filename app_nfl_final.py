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

st.set_page_config(page_title="NFL Analytics Real V2", layout="wide", page_icon="🏈")
st.title("🏈 NFL Analytics Real V2")
st.caption("Sin cuotas inventadas. Sin líneas ficticias. Dato faltante = NO BET. Auto-bets habilitadas solo en mercados con respaldo OOS.")

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
    "moneyline_winner_accuracy": 60.89, "home_baseline": 53.14,
    "spread_ml_accuracy": 53.14, "xgb_accuracy": 51.13, "xgb_brier": 0.2525,
    "ou_accuracy": 47.43, "ou_2025_roi": -3.19,
}


def num(v):
    try:
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def american_to_decimal(v):
    x=num(v)
    if x is None or x == 0: return None
    return 1 + (x/100 if x > 0 else 100/abs(x))


def no_vig(a,b):
    da,db=american_to_decimal(a),american_to_decimal(b)
    if da is None or db is None: return None,None
    ia,ib=1/da,1/db; s=ia+ib
    return ia/s, ib/s


def normal_gt(mean, threshold, sigma):
    if mean is None or threshold is None or sigma is None or sigma <= 0: return None
    z=(float(mean)-float(threshold))/float(sigma)
    return 50*(1+math.erf(z/math.sqrt(2)))


def two_way(a,b):
    if a is None or b is None: return None,None
    s=float(a)+float(b)
    if s <= 0: return None,None
    return 100*float(a)/s,100*float(b)/s


def value_metrics(pct, odd, mkt):
    d=american_to_decimal(odd)
    if d is None or mkt is None or pct is None: return None
    return {"ev":((pct/100)*d-1)*100,"edge":(pct/100-mkt)*100,"decimal":d}


def moneyline_candidate(partido, apuesta, probs, odd_self, odd_other):
    probs=[float(p) for p in probs if p is not None]
    if len(probs) < 3: return None
    disagreement=max(probs)-min(probs)
    if disagreement > 15: return None
    p=float(np.mean(probs))
    mkt,_=no_vig(odd_self,odd_other)
    vm=value_metrics(p,odd_self,mkt)
    if vm is None: return None
    if p < 54 or vm["edge"] < 3 or vm["ev"] < 3: return None
    return {
        "Partido":partido,"Mercado":"Moneyline","Apuesta":apuesta,"Prob consenso":round(p,1),
        "Modelos":f"{min(probs):.1f}-{max(probs):.1f}%","Desacuerdo pp":round(disagreement,1),
        "Momio real":int(float(odd_self)),"Edge no-vig pp":round(vm["edge"],2),"EV %":round(vm["ev"],2),
        "_score":1.5*vm["edge"]+vm["ev"]-0.2*disagreement,
    }


@st.cache_data(ttl=3600)
def cargar_historico():
    return pd.read_csv("data/historico_nfl_games.csv"), pd.read_csv("data/historico_nfl_qbs.csv")


@st.cache_data(ttl=300)
def cargar_schedule(season):
    try: return nfl.import_schedules([int(season)])
    except Exception: return pd.DataFrame()


@st.cache_resource
def motores(df):
    ml=PredictorNFL_ML(); ml_ok=ml.entrenar(df)
    xgb=PredictorXGBoostSpread(); xgb_ok=xgb.entrenar(df)
    elo=MotorELONFL(); elo.actualizar_ratings(df)
    return ml,ml_ok,xgb,xgb_ok,elo


@st.cache_data(ttl=1800)
def forecast_kickoff(team,gameday,gametime):
    info=ESTADIOS.get(team)
    if not info: return None,None,False,"Sin coordenadas"
    lat,lon,dome=info
    if dome: return None,None,True,"Domo"
    try:
        date=pd.to_datetime(gameday).date(); hhmm=str(gametime or "13:00")[:5]
        et=dt.datetime.combine(date,dt.time.fromisoformat(hhmm),tzinfo=ZoneInfo("America/New_York"))
        utc=et.astimezone(ZoneInfo("UTC"))
        params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m,wind_speed_10m","temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"UTC","start_date":utc.date().isoformat(),"end_date":utc.date().isoformat()}
        data=requests.get("https://api.open-meteo.com/v1/forecast",params=params,timeout=8).json()
        times=pd.to_datetime(data.get("hourly",{}).get("time",[]),utc=True)
        if not len(times): return None,None,False,"Forecast no disponible"
        idx=int(np.argmin(np.abs(times-pd.Timestamp(utc))))
        t=num(data["hourly"]["temperature_2m"][idx]); w=num(data["hourly"]["wind_speed_10m"][idx])
        return t,w,False,f"{t:.1f}°F, {w:.1f} mph al kickoff"
    except Exception:
        return None,None,False,"Forecast no disponible"


df_games,df_qbs=cargar_historico()
ml_engine,ml_ok,xgb_engine,xgb_ok,elo_engine=motores(df_games)

with st.expander("✅ Validación fuera de muestra",expanded=True):
    c1,c2,c3=st.columns(3)
    c1.metric("Moneyline 2025",f"{VALIDATION['moneyline_2025_wins']}/{VALIDATION['moneyline_2025_picks']}",f"ROI +{VALIDATION['moneyline_2025_roi']}%")
    c2.metric("Spread","NO AUTO BET",f"ML {VALIDATION['spread_ml_accuracy']}% / XGB {VALIDATION['xgb_accuracy']}%")
    c3.metric("O/U","NO AUTO BET",f"2025 ROI {VALIDATION['ou_2025_roi']}%")
    st.caption("Moneyline mostró señal positiva en 2025 OOS. Spread y Totales permanecen diagnósticos hasta demostrar ROI positivo en holdout real.")

scan_tab,qb_tab,elo_tab=st.tabs(["🤖 Scanner","🎯 QB Props","📈 ELO"])

with scan_tab:
    a,b,c=st.columns(3)
    season=a.number_input("Temporada",2021,2030,2026,1)
    week=b.number_input("Semana",1,22,1,1)
    top_n=c.slider("Máximo de picks Moneyline",1,10,3)
    if st.button("Analizar jornada",type="primary"):
        sched=cargar_schedule(season)
        if sched.empty:
            st.error("No se pudo obtener el schedule real.")
        else:
            games=sched[sched["week"]==int(week)].copy()
            if "game_type" in games: games=games[games["game_type"].isin(["REG","POST","WC","DIV","CON","SB"])]
            if "home_score" in games:
                future=games[games["home_score"].isna()]
                if not future.empty: games=future
            picks=[]; diag=[]
            for _,g in games.iterrows():
                home,away=g.get("home_team"),g.get("away_team")
                if not home or not away: continue
                partido=f"{away} @ {home}"
                hm,am=num(g.get("home_moneyline")),num(g.get("away_moneyline"))
                line=num(g.get("total_line")); spread=num(g.get("spread_line"))
                ho,ao=num(g.get("home_spread_odds")),num(g.get("away_spread_odds"))
                oo,uo=num(g.get("over_odds")),num(g.get("under_odds"))
                hr,ar=num(g.get("home_rest")),num(g.get("away_rest"))
                temp,wind,dome,wmsg=forecast_kickoff(home,g.get("gameday"),g.get("gametime"))
                ml=ml_engine.predecir_contexto(week,home,away,temp,wind,dome,hr,ar) if ml_ok else None
                emp=simular_nfl_montecarlo(home,away,df_games,line,spread)
                row={"Partido":partido,"Clima kickoff":wmsg,"ML odds H/A":f"{hm}/{am}","Spread":spread,"Total":line}
                if not ml or not emp.get("Disponible"):
                    row["Estado"]="Sin datos/modelos suficientes"; diag.append(row); continue
                sigma_m=ml.get("Sigma_Margen_OOS"); sigma_t=ml.get("Sigma_Total_OOS")
                pml_h=normal_gt(ml.get("ML_Margen_Local_Esperado"),0,sigma_m)
                pe_h,pe_a=two_way(emp["Moneyline"].get("Gana Local"),emp["Moneyline"].get("Gana Visita"))
                pelo=100*elo_engine.calcular_probabilidad_elo(elo_engine.ratings.get(home,1500),elo_engine.ratings.get(away,1500))
                if hm is not None and am is not None and pml_h is not None:
                    ch=moneyline_candidate(partido,f"{home} ML",[pelo,pml_h,pe_h],hm,am)
                    ca=moneyline_candidate(partido,f"{away} ML",[100-pelo,100-pml_h,pe_a],am,hm)
                    if ch: picks.append(ch)
                    if ca: picks.append(ca)
                # Spread diagnostics only
                if spread is not None and sigma_m:
                    ml_cov=normal_gt(ml.get("ML_Margen_Local_Esperado"),spread,sigma_m)
                    e_h,e_a=two_way(emp["Spread"].get("Cubre Local"),emp["Spread"].get("Cubre Visita"))
                    xh=xgb_engine.predecir_probabilidad_cover(week,spread,line,home,away,hr,ar) if xgb_ok and line is not None else None
                    row["Spread probs ML/Emp/XGB"]=f"{None if ml_cov is None else round(ml_cov,1)} / {None if e_h is None else round(e_h,1)} / {None if xh is None else round(xh,1)}"
                    row["Spread estado"]="NO AUTO BET — OOS insuficiente"
                # Total diagnostics only
                weather_ok=dome or (temp is not None and wind is not None)
                if line is not None and sigma_t and weather_ok:
                    mo=normal_gt(ml.get("ML_Puntos_Totales_Esperados"),line,sigma_t)
                    eo,eu=two_way(emp["Over_Under"].get("Prob Over"),emp["Over_Under"].get("Prob Under"))
                    row["O/U probs Over ML/Emp"]=f"{None if mo is None else round(mo,1)} / {None if eo is None else round(eo,1)}"
                    row["O/U estado"]="NO AUTO BET — ROI 2025 negativo"
                row["Estado"]="Analizado"; diag.append(row)
            if picks:
                out=pd.DataFrame(picks).sort_values("_score",ascending=False).head(top_n).drop(columns=["_score"])
                st.success(f"{len(out)} Moneyline picks superan consenso + no-vig + EV.")
                st.dataframe(out,use_container_width=True,hide_index=True)
            else:
                st.info("No hay Moneyline con valor suficiente hoy.")
            if diag:
                st.markdown("### Diagnóstico completo (Spread/O-U no generan picks)")
                st.dataframe(pd.DataFrame(diag),use_container_width=True,hide_index=True)

with qb_tab:
    st.info("QB props son manuales hasta disponer de líneas/odds de props en una fuente real conectada.")
    if not df_qbs.empty:
        qb=st.selectbox("Quarterback",sorted(df_qbs["player_name"].dropna().astype(str).unique()))
        line=st.number_input("Línea REAL passing yards",100.0,450.0,245.5,0.5)
        x,y=st.columns(2)
        over=x.number_input("Momio real Over (0=sin dato)",value=0,step=5)
        under=y.number_input("Momio real Under (0=sin dato)",value=0,step=5)
        if st.button("Analizar prop"):
            res=PredictorYardasQB(df_qbs).proyectar_yardas_qb(qb,line)
            st.json(res)
            if "error" not in res and over!=0 and under!=0:
                po,pu=no_vig(over,under)
                vo=value_metrics(res["Prob_Over_Yardas"],over,po); vu=value_metrics(res["Prob_Under_Yardas"],under,pu)
                st.write({"Over edge pp":round(vo["edge"],2),"Over EV%":round(vo["ev"],2),"Under edge pp":round(vu["edge"],2),"Under EV%":round(vu["ev"],2)})

with elo_tab:
    rank=pd.DataFrame(elo_engine.obtener_power_ranking(),columns=["Equipo","ELO"]); rank["ELO"]=rank["ELO"].round(1)
    st.dataframe(rank,use_container_width=True,hide_index=True)
