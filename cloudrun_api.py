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
from modules.nfl_google_sheets import sync_bets
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo

app = FastAPI(title="NFL Analytics API", version="3.2")
MODEL_CACHE = {}
DEFAULT_BANKROLL = 5000.0
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.05


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


def kelly_stake(probability_pct, odd, bankroll):
    dec = american_to_decimal(odd)
    p = num(probability_pct)
    bank = num(bankroll)
    if dec is None or p is None or bank is None or bank <= 0:
        return 0.0, 0.0, False
    b = dec - 1.0
    if b <= 0:
        return 0.0, 0.0, False
    p = min(max(p / 100.0, 0.0), 1.0)
    q = 1.0 - p
    full_kelly = max(0.0, (b * p - q) / b)
    raw_fraction = full_kelly * KELLY_FRACTION
    capped_fraction = min(raw_fraction, MAX_STAKE_FRACTION)
    capped = raw_fraction > capped_fraction + 1e-12
    return round(capped_fraction * 100.0, 2), round(bank * capped_fraction, 2), capped


def candidate(game, pick, primary_prob, support_probs, odd_self, odd_other, bankroll=DEFAULT_BANKROLL):
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
    is_bet = odd < 0
    kelly_pct, stake, kelly_capped = kelly_stake(p, odd, bankroll)
    return {
        "game": game,
        "pick": pick,
        "probability": round(p, 1),
        "odds": int(odd),
        "edge": round(edge, 2),
        "ev": round(ev, 2),
        "kelly": kelly_pct,
        "stake": stake if is_bet else 0.0,
        "kelly_capped": kelly_capped,
        "disagreement": round(disagreement, 1),
        "action": "BET" if is_bet else "LEAN",
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
    return {"status": "ok", "service": "NFL Analytics Cloud Run", "version": "3.2"}


@app.get("/api/scan/{season}/{week}")
def scan(season: int, week: int, bankroll: float = DEFAULT_BANKROLL):
    try:
        if bankroll <= 0:
            raise HTTPException(status_code=400, detail="El bankroll debe ser mayor que 0")
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
                ch = candidate(game, f"{home} ML", p_h, [elo_h, e_h], hm, am, bankroll)
                ca = candidate(game, f"{away} ML", p_a, [100 - elo_h, e_a], am, hm, bankroll)
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
        sheet_sync = sync_bets(bets, season, week, bankroll)
        return {
            "season": season,
            "week": week,
            "bankroll": round(bankroll, 2),
            "kelly_policy": "1/4 Kelly, máximo 5% del bankroll por BET",
            "bets": bets,
            "leans": leans,
            "diagnostics": diagnostics,
            "sheet_sync": sheet_sync,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>NFL Analytics</title><style>body{font-family:Arial;max-width:900px;margin:40px auto;padding:0 18px;background:#0b1020;color:#fff}input,button{padding:12px;margin:6px;border-radius:8px;border:0}button{cursor:pointer;font-weight:700}.card{background:#151d33;padding:18px;border-radius:12px;margin-top:15px}.muted{color:#aeb8d0;font-size:.92rem}.stake{margin-top:8px;font-weight:700}.lean{border:1px solid #59647d}details{margin-top:18px}summary{cursor:pointer;font-weight:700}</style></head><body><h1>🏈 NFL Analytics</h1><p>Cloud Run + FastAPI</p><input id='s' type='number' value='2026' min='2021' max='2030'><input id='w' type='number' value='1' min='1' max='22'><input id='b' type='number' value='5000' min='100' step='500'><button onclick='go()'>Escanear jornada</button><div id='out' class='card'>Listo.</div><script>function money(v){return Number(v||0).toLocaleString('es-MX',{minimumFractionDigits:2,maximumFractionDigits:2})}function card(p,autoBet){const cap=p.kelly_capped?' · tope 5% aplicado':'';const stake=autoBet?`<div class="stake">Kelly 1/4: ${p.kelly}% · Apostar $${money(p.stake)}${cap}</div>`:`<div class="stake">Kelly 1/4 teórico: ${p.kelly}% · Apostar $0.00</div><div class="muted">LEAN — NO AUTO BET</div>`;return `<div class="card ${autoBet?'':'lean'}"><b>${p.pick}</b><br><span class="muted">${p.game}</span><br>Prob ${p.probability}% · Edge ${p.edge} pp · EV ${p.ev}% · Momio ${p.odds}${stake}</div>`}async function go(){const o=document.getElementById('out');o.innerHTML='Analizando...';try{const r=await fetch(`/api/scan/${s.value}/${w.value}?bankroll=${encodeURIComponent(b.value)}`);const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');let h=`<h2>Recomendaciones</h2><div class="muted">Bankroll $${money(j.bankroll)} · ${j.kelly_policy}</div>`;const ss=j.sheet_sync||{};h+=ss.ok?`<div class="muted">Sheet NFL_Picks: ${ss.inserted||0} nuevas · ${ss.updated||0} actualizadas</div>`:`<div class="muted">Sheet NFL_Picks: no guardó · ${ss.message||'error'}</div>`;if(j.bets.length===0)h+='<p>No hay BET robusto.</p>';for(const p of j.bets)h+=card(p,true);if(j.leans.length){h+=`<details><summary>LEAN (${j.leans.length}) — señales secundarias</summary>`;for(const p of j.leans)h+=card(p,false);h+='</details>'}o.innerHTML=h}catch(e){o.innerHTML='<b>Error:</b> '+e.message}}</script></body></html>"""
