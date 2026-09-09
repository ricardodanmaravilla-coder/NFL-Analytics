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
from modules.nfl_google_sheets import settle_pending, sync_bets
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo
from modules.nfl_therundown_odds import configured as therundown_configured, get_moneyline
from modules.nfl_weather import forecast_kickoff

app = FastAPI(title="NFL Analytics API", version="3.7")
MODEL_CACHE = {}
DEFAULT_BANKROLL = 5000.0
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.05
MIN_PROBABILITY = 56.0
MAX_DISAGREEMENT = 15.0


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


def _build_candidate(game, pick, probability, support_probs, odd_self, odd_other, bankroll,
                     auto_bet, market, line=None, book=None, source=None, fetched_at=None):
    p = num(probability)
    supports = [num(x) for x in support_probs]
    supports = [x for x in supports if x is not None]
    if p is None or not supports:
        return None
    probs = [p] + supports
    if max(probs) - min(probs) > MAX_DISAGREEMENT:
        return None
    primary_side = p >= 50.0
    if any((x >= 50.0) != primary_side for x in supports):
        return None

    mkt, _ = no_vig(odd_self, odd_other)
    dec = american_to_decimal(odd_self)
    odd = num(odd_self)
    if mkt is None or dec is None or odd is None:
        return None
    edge = (p / 100.0 - mkt) * 100.0
    ev = ((p / 100.0) * dec - 1.0) * 100.0
    disagreement = max(probs) - min(probs)
    if p < MIN_PROBABILITY or edge < 3.0 or ev < 3.0:
        return None

    kelly_pct, stake, kelly_capped = kelly_stake(p, odd, bankroll)
    action = "BET" if auto_bet else "LEAN"
    return {
        "game": game,
        "pick": pick,
        "market": market,
        "line": line,
        "probability": round(p, 1),
        "odds": int(odd),
        "edge": round(edge, 2),
        "ev": round(ev, 2),
        "kelly": kelly_pct,
        "stake": stake if auto_bet else 0.0,
        "kelly_capped": kelly_capped,
        "disagreement": round(disagreement, 1),
        "action": action,
        "score": round(1.5 * edge + ev - 0.3 * disagreement, 3),
        "book": book,
        "odds_source": source,
        "odds_fetched_at": fetched_at,
    }


def candidate(game, pick, primary_prob, support_probs, odd_self, odd_other, bankroll=DEFAULT_BANKROLL,
              book=None, source=None, fetched_at=None):
    """Moneyline: conserva consenso fuerte y sólo favoritos como auto-BET."""
    p = primary_with_agreement(primary_prob, support_probs, max_disagreement=MAX_DISAGREEMENT)
    if p is None:
        return None
    odd = num(odd_self)
    return _build_candidate(
        game, pick, p, support_probs, odd_self, odd_other, bankroll,
        auto_bet=bool(odd is not None and odd < 0), market="ML",
        book=book, source=source, fetched_at=fetched_at,
    )


def market_candidate(game, pick, market, line, primary_prob, mc_prob, odd_self, odd_other,
                     bankroll=DEFAULT_BANKROLL, book=None, source=None, fetched_at=None):
    """Spread/Total: ML calibrado es primario y Monte Carlo empírico es guardrail.

    A diferencia de Moneyline no existe concepto favorito/underdog para bloquear un
    auto-BET. Si supera 56%, Edge 3 pp, EV 3% y desacuerdo <=15 pp, es BET.
    """
    return _build_candidate(
        game, pick, primary_prob, [mc_prob], odd_self, odd_other, bankroll,
        auto_bet=True, market=market, line=line,
        book=book, source=source, fetched_at=fetched_at,
    )


def fmt_line(v):
    x = num(v)
    if x is None:
        return "?"
    return f"{x:+g}"


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
        raise RuntimeError("No hay histórico suficiente para entrenar modelos NFL")
    elo = MotorELONFL()
    elo.actualizar_ratings(past_games)
    MODEL_CACHE[key] = (ml, elo, past_games)
    return MODEL_CACHE[key]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "NFL Analytics Cloud Run",
        "version": "3.7",
        "kickoff_weather": True,
        "live_odds_provider": "TheRundown",
        "markets": ["moneyline", "spread", "total"],
        "therundown_configured": therundown_configured(),
        "min_probability": MIN_PROBABILITY,
        "current_odds_policy": "TheRundown only; nflverse lines are historical/backtest only",
    }


@app.get("/api/settle")
def settle():
    return settle_pending()


@app.get("/api/scan/{season}/{week}")
def scan(season: int, week: int, bankroll: float = DEFAULT_BANKROLL):
    try:
        if bankroll <= 0:
            raise HTTPException(status_code=400, detail="El bankroll debe ser mayor que 0")

        settlement = settle_pending()
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
                quote = get_moneyline(home, away, g.get("gameday"))
                if not quote:
                    diagnostics.append({"game": game, "status": "NO BET - sin mercados reales de TheRundown", "odds_source": "TheRundown"})
                    continue
                if str(g.get("location", "")).strip().lower() == "neutral":
                    diagnostics.append({"game": game, "status": "NO BET - sede neutral"})
                    continue

                hm, am = num(quote.get("home_moneyline")), num(quote.get("away_moneyline"))
                hs, aso = num(quote.get("home_spread")), num(quote.get("away_spread"))
                hso, aso_odds = num(quote.get("home_spread_odds")), num(quote.get("away_spread_odds"))
                total_line = num(quote.get("total_line"))
                over_odds, under_odds = num(quote.get("over_odds")), num(quote.get("under_odds"))

                hr, ar = num(g.get("home_rest")), num(g.get("away_rest"))
                temp, wind, dome, weather_msg = forecast_kickoff(home, g.get("gameday"), g.get("gametime"), g.get("roof"))
                pred = ml.predecir_contexto(week, home, away, temp, wind, dome, hr, ar)
                spread_threshold = -hs if hs is not None else None
                emp = simular_nfl_montecarlo(home, away, past_games, total_line, spread_threshold)
                if not pred or not emp.get("Disponible"):
                    diagnostics.append({"game": game, "status": "Sin datos suficientes", "weather": weather_msg})
                    continue

                source = quote.get("source")
                fetched_at = quote.get("fetched_at")

                # MONEYLINE: modelo de margen calibrado + Elo + Monte Carlo.
                if hm is not None and am is not None:
                    p_h, p_a = empirical_residual_two_way(pred.get("ML_Margen_Local_Esperado"), 0.0, ml.residuales_margen)
                    e_h, e_a = two_way(emp["Moneyline"].get("Gana Local"), emp["Moneyline"].get("Gana Visita"))
                    elo_h = 100 * elo.calcular_probabilidad_elo(elo.ratings.get(home, 1500), elo.ratings.get(away, 1500))
                    meta_ml = {"book": quote.get("book"), "source": source, "fetched_at": fetched_at}
                    ch = candidate(game, f"{home} ML", p_h, [elo_h, e_h], hm, am, bankroll, **meta_ml)
                    ca = candidate(game, f"{away} ML", p_a, [100 - elo_h, e_a], am, hm, bankroll, **meta_ml)
                    if ch: picks.append(ch)
                    if ca: picks.append(ca)

                # SPREAD/HANDICAP: TheRundown usa signo de apuesta (-3.5 favorito).
                # El motor trabaja con umbral de margen local positivo, por eso threshold=-home_spread.
                if None not in (hs, aso, hso, aso_odds, spread_threshold):
                    sp_h, sp_a = empirical_residual_two_way(
                        pred.get("ML_Margen_Local_Esperado"), spread_threshold, ml.residuales_margen
                    )
                    mc_h = num(emp.get("Spread", {}).get("Cubre Local"))
                    mc_a = num(emp.get("Spread", {}).get("Cubre Visita"))
                    spread_book = quote.get("spread_book") or quote.get("book")
                    csh = market_candidate(game, f"{home} {fmt_line(hs)}", "SPREAD", hs, sp_h, mc_h,
                                           hso, aso_odds, bankroll, spread_book, source, fetched_at)
                    csa = market_candidate(game, f"{away} {fmt_line(aso)}", "SPREAD", aso, sp_a, mc_a,
                                           aso_odds, hso, bankroll, spread_book, source, fetched_at)
                    if csh: picks.append(csh)
                    if csa: picks.append(csa)

                # TOTAL: bosque de puntos calibrado OOS + distribución empírica Monte Carlo.
                if None not in (total_line, over_odds, under_odds):
                    p_over, p_under = empirical_residual_two_way(
                        pred.get("ML_Puntos_Totales_Esperados"), total_line, ml.residuales_total
                    )
                    mc_over = num(emp.get("Over_Under", {}).get("Prob Over"))
                    mc_under = num(emp.get("Over_Under", {}).get("Prob Under"))
                    total_book = quote.get("total_book") or quote.get("book")
                    cov = market_candidate(game, f"Over {total_line:g}", "TOTAL", total_line, p_over, mc_over,
                                           over_odds, under_odds, bankroll, total_book, source, fetched_at)
                    cun = market_candidate(game, f"Under {total_line:g}", "TOTAL", total_line, p_under, mc_under,
                                           under_odds, over_odds, bankroll, total_book, source, fetched_at)
                    if cov: picks.append(cov)
                    if cun: picks.append(cun)

                diagnostics.append({
                    "game": game,
                    "status": "Analizado ML + spread + total con líneas reales disponibles",
                    "book": quote.get("book"),
                    "home_moneyline": int(hm) if hm is not None else None,
                    "away_moneyline": int(am) if am is not None else None,
                    "home_spread": hs,
                    "home_spread_odds": int(hso) if hso is not None else None,
                    "away_spread": aso,
                    "away_spread_odds": int(aso_odds) if aso_odds is not None else None,
                    "spread_book": quote.get("spread_book"),
                    "total_line": total_line,
                    "over_odds": int(over_odds) if over_odds is not None else None,
                    "under_odds": int(under_odds) if under_odds is not None else None,
                    "total_book": quote.get("total_book"),
                    "ml_margin_projection": pred.get("ML_Margen_Local_Esperado"),
                    "ml_total_projection": pred.get("ML_Puntos_Totales_Esperados"),
                    "mc_total_projection": emp.get("Proyeccion_Score", {}).get("Total_Proyectado"),
                    "odds_source": source,
                    "odds_fetched_at": fetched_at,
                    "weather": weather_msg,
                    "temp_f": temp,
                    "wind_mph": wind,
                    "dome": dome,
                })
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
            "min_probability": MIN_PROBABILITY,
            "kelly_policy": "1/4 Kelly, máximo 5% del bankroll por BET",
            "markets": ["ML", "SPREAD", "TOTAL"],
            "market_policy": "ML: ML+Elo+MC; Spread/Total: ML calibrado+MC; max desacuerdo 15 pp; Edge>=3; EV>=3",
            "odds_policy": "TheRundown live/delayed feed only; no nflverse fallback for current prices",
            "bets": bets,
            "leans": leans,
            "diagnostics": diagnostics,
            "sheet_sync": sheet_sync,
            "settlement": settlement,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>NFL Analytics</title></head><body><h1>NFL Analytics API</h1><p>Use Cloud Run production entrypoint.</p></body></html>"""
