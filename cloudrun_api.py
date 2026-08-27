import os
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import nfl_data_py as nfl
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from modules.nfl_calibration import empirical_residual_two_way, historico_antes, primary_with_agreement
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo

app = FastAPI(title="NFL Analytics API", version="3.0")
MODEL_CACHE = {}


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


def two_way(a, b):
    if a is None or b is None:
        return None, None
    total = float(a) + float(b)
    return (100 * float(a) / total, 100 * float(b) / total) if total > 0 else (None, None)


def candidate(game, pick, primary_prob, support_probs, odd_self, odd_other):
    p = primary_with_agreement(primary_prob, support_probs, max_disagreement=15.0)
    if p is None:
        return None
    mkt, _ = no_vig(odd_self, odd_other)
    dec = american_to_decimal(odd_self)
    odd = num(odd_self)
    if mkt is None or dec is None or odd is None:
        return None
    edge = (p / 100 - mkt) * 100
    ev = ((p / 100) * dec - 1) * 100
    probs = [float(p)] + [float(x) for x in support_probs if x is not None]
    disagreement = max(probs) - min(probs)
    if p < 54 or edge < 3 or ev < 3:
        return None
    return {
        "game": game,
        "pick": pick,
        "probability": round(p, 1),
        "odds": int(odd),
        "edge": round(edge, 2),
        "ev": round(ev, 2),
        "disagreement": round(disagreement, 1),
        "action": "BET" if odd < 0 else "LEAN",
        "score": round(1.5 * edge + ev - 0.3 * disagreement, 3),
    }


@lru_cache(maxsize=1)
def load_history():
    games = pd.read_csv("data/historico_nfl_games.csv")
    pbp_path = "data/historico_nfl_pbp_team_game.csv"
    pbp = pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    return games, pbp


def get_models(season, week):
    key = (int(season), int(week))
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    games, pbp = load_history()
    past_games = historico_antes(games, season, week)
    past_pbp = historico_antes(pbp, season, week) if not pbp.empty else pd.DataFrame()
    ml = MoneylineRuntime()
    if not ml.entrenar(past_games, past_pbp):
        raise RuntimeError("No hay histórico suficiente para entrenar Moneyline")
    elo = MotorELONFL()
    elo.actualizar_ratings(past_games)
    MODEL_CACHE[key] = (ml, elo, past_games)
    return MODEL_CACHE[key]


@app.get("/health")
def health():
    return {"status": "ok", "service": "NFL Analytics Cloud Run"}


@app.get("/api/scan/{season}/{week}")
def scan(season: int, week: int):
    try:
        sched = nfl.import_schedules([season])
        games = sched[sched["week"] == week].copy()
        if "game_type" in games.columns:
            games = games[games["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
        if "home_score" in games.columns:
            future = games[games["home_score"].isna()]
            if not future.empty:
                games = future
        ml, elo, past_games = get_models(season, week)
        picks, diagnostics = [], []
        for _, g in games.iterrows():
            try:
                home, away = g.get("home_team"), g.get("away_team")
                if not home or not away:
                    continue
                game = f"{away} @ {home}"
                hm, am = num(g.get("home_moneyline")), num(g.get("away_moneyline"))
                if hm is None or am is None:
                    diagnostics.append({"game": game, "status": "Sin momios Moneyline"})
                    continue
                if str(g.get("location", "")).strip().lower() == "neutral":
                    diagnostics.append({"game": game, "status": "NO BET - sede neutral"})
                    continue
                hr, ar = num(g.get("home_rest")), num(g.get("away_rest"))
                pred = ml.predecir_contexto(week, home, away, None, None, False, hr, ar)
                emp = simular_nfl_montecarlo(home, away, past_games, num(g.get("total_line")), num(g.get("spread_line")))
                if not pred or not emp.get("Disponible"):
                    diagnostics.append({"game": game, "status": "Sin datos suficientes"})
                    continue
                p_h, p_a = empirical_residual_two_way(pred.get("ML_Margen_Local_Esperado"), 0.0, ml.residuales_margen)
                e_h, e_a = two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
                elo_h = 100 * elo.calcular_probabilidad_elo(elo.ratings.get(home, 1500), elo.ratings.get(away, 1500))
                ch = candidate(game, f"{home} ML", p_h, [elo_h, e_h], hm, am)
                ca = candidate(game, f"{away} ML", p_a, [100 - elo_h, e_a], am, hm)
                if ch:
                    picks.append(ch)
                if ca:
                    picks.append(ca)
                diagnostics.append({"game": game, "status": "Analizado"})
            except Exception as exc:
                diagnostics.append({"game": str(g.get("game_id", "?")), "status": f"Error: {type(exc).__name__}"})
        picks = sorted(picks, key=lambda x: x["score"], reverse=True)
        bets = [p for p in picks if p["action"] == "BET"]
        leans = [p for p in picks if p["action"] == "LEAN"]
        return {"season": season, "week": week, "bets": bets, "leans": leans, "diagnostics": diagnostics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>NFL Analytics</title><style>body{font-family:Arial;max-width:900px;margin:40px auto;padding:0 18px;background:#0b1020;color:#fff}input,button{padding:12px;margin:6px;border-radius:8px;border:0}button{cursor:pointer;font-weight:700}.card{background:#151d33;padding:18px;border-radius:12px;margin-top:15px}pre{white-space:pre-wrap}</style></head><body><h1>🏈 NFL Analytics</h1><p>Cloud Run + FastAPI</p><input id='s' type='number' value='2026'><input id='w' type='number' value='1'><button onclick='go()'>Escanear jornada</button><div id='out' class='card'>Listo.</div><script>async function go(){const o=document.getElementById('out');o.innerHTML='Analizando...';try{const r=await fetch(`/api/scan/${s.value}/${w.value}`);const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');let h='<h2>Recomendaciones</h2>';if(j.bets.length===0)h+='<p>No hay BET robusto.</p>';for(const p of j.bets){h+=`<div class="card"><b>${p.pick}</b><br>${p.game}<br>Prob ${p.probability}% · Edge ${p.edge} pp · EV ${p.ev}% · Momio ${p.odds}</div>`}if(j.leans.length)h+=`<details><summary>LEAN (${j.leans.length})</summary><pre>${JSON.stringify(j.leans,null,2)}</pre></details>`;o.innerHTML=h}catch(e){o.innerHTML='<b>Error:</b> '+e.message}}</script></body></html>"""
