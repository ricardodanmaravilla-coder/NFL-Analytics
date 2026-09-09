"""Production Cloud Run entrypoint with fresh Parquet-first history.

The HTTP routes remain in `cloudrun_api`. This wrapper patches only data/model
loading before the first request so a long-lived Cloud Run revision can consume
new weekly history without requiring a redeploy. Remote validated Parquet is
preferred; bundled local Parquet/CSV remains an atomic fallback.
"""
from functools import lru_cache
import os
import threading
import time

import pandas as pd

import cloudrun_api as base
from modules.nfl_calibration import historico_antes
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_production_history import load_production_history
from modules.nfl_therundown_odds import diagnose_date

HISTORY_TTL_SECONDS = max(900, int(os.getenv("NFL_HISTORY_TTL_SECONDS", "3600")))
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = {}
_LAST_BUCKET = None


def _bucket() -> int:
    return int(time.time() // HISTORY_TTL_SECONDS)


@lru_cache(maxsize=2)
def _history_for_bucket(bucket: int):
    games, pbp, source = load_production_history(prefer_remote=True)
    base.PBP_STORAGE = source
    return games, pbp


def load_history_parquet_first():
    games, pbp = _history_for_bucket(_bucket())
    return games.copy(), pbp.copy()


def get_models_fresh(season, week):
    """Cache trained models only inside the same history freshness bucket."""
    global _LAST_BUCKET
    bucket = _bucket()
    key = (int(season), int(week), bucket)
    with _CACHE_LOCK:
        if _LAST_BUCKET != bucket:
            _MODEL_CACHE.clear()
            _history_for_bucket.cache_clear()
            _LAST_BUCKET = bucket
        cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    games, pbp = load_history_parquet_first()
    past_games = historico_antes(games, season, week)
    past_pbp = historico_antes(pbp, season, week) if not pbp.empty else pd.DataFrame()
    ml = MoneylineRuntime()
    if not ml.entrenar(past_games, past_pbp):
        raise RuntimeError("No hay histórico suficiente para entrenar Moneyline")
    elo = MotorELONFL()
    elo.actualizar_ratings(past_games)
    result = (ml, elo, past_games)
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = result
    return result


base.load_history = load_history_parquet_first
base.get_models = get_models_fresh
base.MODEL_CACHE.clear()
base.PBP_STORAGE = "UNINITIALIZED"
base.HISTORY_TTL_SECONDS = HISTORY_TTL_SECONDS

app = base.app


@app.get("/api/storage")
def storage_status():
    games, pbp = load_history_parquet_first()
    seasons = sorted(pd.to_numeric(games["season"], errors="coerce").dropna().astype(int).unique().tolist())
    return {
        "history_source": base.PBP_STORAGE,
        "history_ttl_seconds": HISTORY_TTL_SECONDS,
        "games_rows": int(len(games)),
        "pbp_rows": int(len(pbp)),
        "seasons": seasons,
    }


@app.get("/api/odds/diagnostics/{gameday}")
def odds_diagnostics(gameday: str):
    """Inspect TheRundown availability without ever exposing the API key."""
    return diagnose_date(gameday)


def _week_games(season: int, week: int):
    sched = base.nfl.import_schedules([season])
    games = sched[sched["week"] == week].copy()
    if "game_type" in games.columns:
        games = games[games["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    return games


@app.get("/api/slate/{season}/{week}")
def slate(season: int, week: int):
    """Return the full week immediately with current TheRundown moneylines.

    This endpoint intentionally does not train ML, run Monte Carlo, fetch weather,
    settle the Sheet, or calculate Kelly. It keeps the schedule/price view fast
    and independent from the expensive recommendation scan.
    """
    try:
        games = _week_games(season, week)
        rows = []
        for _, g in games.iterrows():
            home, away = g.get("home_team"), g.get("away_team")
            if not home or not away:
                continue
            quote = base.get_moneyline(home, away, g.get("gameday"))
            hm = base.num(quote.get("home_moneyline")) if quote else None
            am = base.num(quote.get("away_moneyline")) if quote else None
            rows.append({
                "game": f"{away} @ {home}",
                "away": away,
                "home": home,
                "gameday": str(g.get("gameday") or ""),
                "gametime": str(g.get("gametime") or ""),
                "away_moneyline": int(am) if am is not None else None,
                "home_moneyline": int(hm) if hm is not None else None,
                "book": quote.get("book") if quote else None,
                "odds_source": quote.get("source") if quote else "TheRundown",
                "odds_fetched_at": quote.get("fetched_at") if quote else None,
                "status": "CUOTA REAL" if hm is not None and am is not None else "CUOTA NO DISPONIBLE",
            })
        return {"season": season, "week": week, "games": rows, "count": len(rows), "odds_provider": "TheRundown"}
    except Exception as exc:
        base.HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


# Replace the legacy root page while preserving every API route.
app.router.routes[:] = [r for r in app.router.routes if not (getattr(r, "path", None) == "/" and "GET" in getattr(r, "methods", set()))]


@app.get("/", response_class=base.HTMLResponse)
def home_slate():
    return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>NFL Analytics</title><style>
body{font-family:Arial;max-width:1100px;margin:32px auto;padding:0 18px;background:#0b1020;color:#fff}input,button{padding:12px;margin:6px;border-radius:8px;border:0}button{cursor:pointer;font-weight:700}.card{background:#151d33;padding:16px;border-radius:12px;margin-top:12px}.muted{color:#aeb8d0;font-size:.92rem}.stake{margin-top:8px;font-weight:700}.lean{border:1px solid #59647d}.game{display:grid;grid-template-columns:1.4fr .8fr .8fr .8fr;gap:12px;align-items:center}.team{font-weight:700}.odd{font-size:1.08rem;font-weight:700}.ok{color:#7ee787}.missing{color:#f2cc60}details{margin-top:18px}summary{cursor:pointer;font-weight:700}@media(max-width:720px){.game{grid-template-columns:1fr 1fr}.game .match{grid-column:1/-1}}
</style></head><body><h1>🏈 NFL Analytics</h1><p class='muted'>Cartelera y cuotas primero; análisis ML + Monte Carlo después.</p><input id='s' type='number' value='2026' min='2021' max='2030'><input id='w' type='number' value='1' min='1' max='22'><input id='b' type='number' value='5000' min='100' step='500'><button onclick='loadSlate()'>Ver jornada y cuotas</button><button onclick='go()'>Analizar jornada</button><button onclick='settleNow()'>Actualizar resultados</button><div id='slate' class='card'>Pulsa “Ver jornada y cuotas”.</div><div id='out' class='card'>Las recomendaciones aparecerán aquí después del análisis.</div><script>
function money(v){return Number(v||0).toLocaleString('es-MX',{minimumFractionDigits:2,maximumFractionDigits:2})}
function momio(v){if(v===null||v===undefined)return '—';return Number(v)>0?'+'+v:String(v)}
function card(p,autoBet){const cap=p.kelly_capped?' · tope 5% aplicado':'';const src=p.book?` · ${p.book}`:'';const stake=autoBet?`<div class="stake">Kelly 1/4: ${p.kelly}% · Apostar $${money(p.stake)}${cap}</div>`:`<div class="stake">Kelly 1/4 teórico: ${p.kelly}% · Apostar $0.00</div><div class="muted">LEAN — NO AUTO BET</div>`;return `<div class="card ${autoBet?'':'lean'}"><b>${p.pick}</b><br><span class="muted">${p.game}</span><br>Prob ${p.probability}% · Edge ${p.edge} pp · EV ${p.ev}% · Momio ${momio(p.odds)}${src}${stake}</div>`}
function settleText(st){if(!st)return '';return st.ok?`<div class="muted">Resultados: ${st.settled||0} liquidados · ${st.pending||0} pendientes</div>`:`<div class="muted">Resultados: error · ${st.message||'desconocido'}</div>`}
async function loadSlate(){const o=document.getElementById('slate');o.innerHTML='Cargando cartelera y cuotas TheRundown...';try{const r=await fetch(`/api/slate/${s.value}/${w.value}`);const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');let h=`<h2>Jornada ${j.week} · ${j.count} partidos</h2><div class="muted">Moneyline actual/demorada: TheRundown. Sin cuota real se muestra como no disponible.</div>`;if(!j.games.length)h+='<p>No se encontraron partidos.</p>';for(const g of j.games){const cls=g.status==='CUOTA REAL'?'ok':'missing';h+=`<div class="card game"><div class="match"><div class="team">${g.away} @ ${g.home}</div><div class="muted">${g.gameday} ${g.gametime||''} · ${g.book||'TheRundown'}</div></div><div><span class="muted">${g.away} ML</span><br><span class="odd">${momio(g.away_moneyline)}</span></div><div><span class="muted">${g.home} ML</span><br><span class="odd">${momio(g.home_moneyline)}</span></div><div class="${cls}">${g.status}</div></div>`}o.innerHTML=h}catch(e){o.innerHTML='<b>Error al cargar jornada:</b> '+e.message}}
async function settleNow(){const o=document.getElementById('out');o.innerHTML='Actualizando resultados...';try{const r=await fetch('/api/settle');const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');o.innerHTML=`<h2>Resultados NFL</h2>${settleText(j)}<div class="muted">GANADA/PERDIDA/PUSH y Profit $ se actualizan en NFL_Picks.</div>`}catch(e){o.innerHTML='<b>Error:</b> '+e.message}}
async function go(){const o=document.getElementById('out');o.innerHTML='Analizando ML + Monte Carlo + Elo + clima... La cartelera de arriba permanece visible.';try{const r=await fetch(`/api/scan/${s.value}/${w.value}?bankroll=${encodeURIComponent(b.value)}`);const j=await r.json();if(!r.ok)throw new Error(j.detail||'Error');let h=`<h2>Recomendaciones</h2><div class="muted">Bankroll $${money(j.bankroll)} · mínimo ${j.min_probability}% · ${j.kelly_policy}</div><div class="muted">Cuotas actuales: TheRundown</div>`;const ss=j.sheet_sync||{};h+=ss.ok?`<div class="muted">Sheet NFL_Picks: ${ss.inserted||0} nuevas · ${ss.updated||0} actualizadas</div>`:`<div class="muted">Sheet NFL_Picks: no guardó · ${ss.message||'error'}</div>`;h+=settleText(j.settlement);if(j.bets.length===0)h+='<p>No hay BET robusto.</p>';for(const p of j.bets)h+=card(p,true);if(j.leans.length){h+=`<details><summary>LEAN (${j.leans.length}) — señales secundarias</summary>`;for(const p of j.leans)h+=card(p,false);h+='</details>'}o.innerHTML=h}catch(e){o.innerHTML='<b>Error:</b> '+e.message}}
loadSlate();
</script></body></html>"""
