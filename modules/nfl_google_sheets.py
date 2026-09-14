from __future__ import annotations

import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Iterable, Mapping
from urllib.parse import quote

SHEET_ID = os.getenv("GOOGLE_SHEETS_ID", "1VsB21QUsQL5EyXu7Sek5WVeNVznECiTuoIMMB4JXno4")
WORKSHEET = os.getenv("GOOGLE_SHEETS_WORKSHEET", "NFL_Picks")
HEADERS = [
    "Fecha", "Temporada", "Semana", "Partido", "Pick", "Probabilidad %", "Momio",
    "Edge pp", "EV %", "Kelly 1/4 %", "Apostar $", "Acción", "Resultado",
    "Profit $", "Fecha cierre", "ID", "Marcador final",
]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# nflverse/nfl_data_py y ESPN no usan siempre la misma abreviatura.
_ESPN_TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LAR": "LA",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _credentials():
    import google.auth
    credentials, project_id = google.auth.default(scopes=SCOPES)
    return credentials, project_id


def _record_id(season: int, week: int, row: Mapping[str, Any]) -> str:
    return f"NFL|{int(season)}|{int(week)}|{_clean(row.get('game'))}|{_clean(row.get('pick'))}"


def _request_json(session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    if not response.ok:
        raise RuntimeError(f"Sheets API {response.status_code}: {response.text[:1000]}")
    return response.json() if response.content else {}


def _profit(stake: float, odds: float, won: bool, push: bool = False) -> float:
    if push:
        return 0.0
    if not won:
        return round(-abs(float(stake)), 2)
    stake = abs(float(stake)); odds = float(odds)
    if odds > 0:
        return round(stake * odds / 100.0, 2)
    if odds < 0:
        return round(stake * 100.0 / abs(odds), 2)
    return 0.0


# Nombre histórico conservado para compatibilidad de tests/imports.
_profit_for_moneyline = _profit


def _format_final_score(away: str, home: str, away_score: float, home_score: float) -> str:
    def _fmt(value: float) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return f"{away} {_fmt(away_score)} - {home} {_fmt(home_score)}"


def sync_bets(bets: Iterable[Mapping[str, Any]], season: int, week: int, bankroll: float,
              sheet_id: str | None = None, worksheet: str | None = None):
    """Inserta snapshots BET inmutables. Re-escanear nunca reescribe una apuesta previa."""
    rows = [dict(x) for x in (bets or [])]
    if not rows:
        return {"ok": True, "inserted": 0, "updated": 0, "skipped_existing": 0, "message": "no bets"}

    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    if not target_sheet_id:
        return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0, "message": "sheet id missing"}

    credentials = None; project_id = None
    try:
        from google.auth.transport.requests import AuthorizedSession
        credentials, project_id = _credentials()
        session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded_range = quote(f"{target_worksheet}!A:Q", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded_range}").get("values", [])

        if not values:
            header_range = quote(f"{target_worksheet}!A1:Q1", safe="")
            _request_json(session, "PUT", f"{base}/{header_range}?valueInputOption=RAW",
                          json={"range": f"{target_worksheet}!A1:Q1", "majorDimension": "ROWS", "values": [HEADERS]})
            values = [HEADERS]
        elif values[0][:len(HEADERS)] != HEADERS:
            return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0,
                    "worksheet": target_worksheet, "message": "header mismatch; existing sheet preserved"}

        existing_ids = {r[15] for r in values[1:] if len(r) >= 16 and r[15]}
        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
        payload = []; skipped = 0; seen = set()
        for bet in rows:
            rec_id = _record_id(season, week, bet)
            if rec_id in existing_ids or rec_id in seen:
                skipped += 1; continue
            seen.add(rec_id)
            payload.append([
                now_mx, int(season), int(week), _clean(bet.get("game")), _clean(bet.get("pick")),
                bet.get("probability", ""), bet.get("odds", ""), bet.get("edge", ""), bet.get("ev", ""),
                bet.get("kelly", ""), bet.get("stake", ""), "BET", "PENDIENTE", "", "", rec_id, "",
            ])

        if payload:
            append_range = quote(f"{target_worksheet}!A:Q", safe="")
            _request_json(session, "POST", f"{base}/{append_range}:append?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS",
                          json={"majorDimension": "ROWS", "values": payload})

        return {"ok": True, "inserted": len(payload), "updated": 0, "skipped_existing": skipped,
                "worksheet": target_worksheet, "message": "saved via Sheets API; existing pick snapshots preserved",
                "credential_type": type(credentials).__name__,
                "service_account_email": getattr(credentials, "service_account_email", None), "adc_project": project_id}
    except Exception as exc:
        return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0,
                "worksheet": target_worksheet, "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                "credential_type": type(credentials).__name__ if credentials is not None else "unresolved",
                "service_account_email": getattr(credentials, "service_account_email", None) if credentials is not None else None,
                "adc_project": project_id}


def _parse_pick(pick: str, home: str, away: str):
    p = str(pick or "").strip()
    if p.endswith(" ML"):
        team = p[:-3].strip()
        if team in {home, away}:
            return {"market": "ML", "team": team}
    m = re.fullmatch(r"(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", p, flags=re.I)
    if m:
        return {"market": "TOTAL", "side": m.group(1).upper(), "line": float(m.group(2))}
    m = re.fullmatch(r"(.+?)\s+([+-][0-9]+(?:\.[0-9]+)?)", p)
    if m and m.group(1).strip() in {home, away}:
        return {"market": "SPREAD", "team": m.group(1).strip(), "line": float(m.group(2))}
    return None


def _grade(parsed, home, away, home_score, away_score):
    market = parsed["market"]
    if market == "ML":
        if home_score == away_score:
            return "PUSH"
        winner = home if home_score > away_score else away
        return "GANADA" if parsed["team"] == winner else "PERDIDA"
    if market == "TOTAL":
        delta = home_score + away_score - parsed["line"]
        if abs(delta) < 1e-9:
            return "PUSH"
        won = delta > 0 if parsed["side"] == "OVER" else delta < 0
        return "GANADA" if won else "PERDIDA"
    if market == "SPREAD":
        margin = (home_score - away_score) if parsed["team"] == home else (away_score - home_score)
        adjusted = margin + parsed["line"]
        if abs(adjusted) < 1e-9:
            return "PUSH"
        return "GANADA" if adjusted > 0 else "PERDIDA"
    return None


def _normalize_team(team: Any) -> str:
    value = str(team or "").strip().upper()
    return _ESPN_TEAM_ALIASES.get(value, value)


def _espn_final_scores(season_week_pairs):
    """Devuelve marcadores FINAL como respaldo cuando nfl_data_py aún no los publica.

    Solo acepta eventos con status.type.completed=true para evitar liquidar partidos en vivo.
    La llave es (season, nfl_week, home, away).
    """
    import requests

    out = {}
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    headers = {"User-Agent": "NFL-Analytics/3.8 settlement"}
    for season, nfl_week in sorted(set(season_week_pairs)):
        # Temporada regular: week 1-18. Para postemporada, ESPN reinicia week en 1.
        if int(nfl_week) <= 18:
            season_type = 2
            espn_week = int(nfl_week)
        else:
            season_type = 3
            espn_week = max(1, int(nfl_week) - 18)
        try:
            response = requests.get(
                url,
                params={"dates": int(season), "seasontype": season_type, "week": espn_week, "limit": 100},
                headers=headers,
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        for event in payload.get("events", []) or []:
            status = ((event.get("status") or {}).get("type") or {})
            if not bool(status.get("completed")):
                continue
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            competitors = competitions[0].get("competitors") or []
            home = away = None
            home_score = away_score = None
            for competitor in competitors:
                team = _normalize_team(((competitor.get("team") or {}).get("abbreviation")))
                try:
                    score = float(competitor.get("score"))
                except Exception:
                    score = None
                if competitor.get("homeAway") == "home":
                    home, home_score = team, score
                elif competitor.get("homeAway") == "away":
                    away, away_score = team, score
            if home and away and home_score is not None and away_score is not None:
                out[(int(season), int(nfl_week), home, away)] = (home_score, away_score)
    return out


def settle_pending(sheet_id: str | None = None, worksheet: str | None = None):
    """Liquida BET pendientes de ML, spread y total usando marcadores finales reales.

    Fuente primaria: nfl_data_py/nflverse. Si el partido existe pero el marcador aún viene
    vacío (o el juego no aparece), usa ESPN scoreboard como respaldo y únicamente liquida
    eventos marcados oficialmente como completed.
    """
    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    credentials = None; project_id = None
    try:
        import pandas as pd
        import nfl_data_py as nfl
        from google.auth.transport.requests import AuthorizedSession

        credentials, project_id = _credentials(); session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded_range = quote(f"{target_worksheet}!A:Q", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded_range}").get("values", [])
        if len(values) <= 1:
            return {"ok": True, "settled": 0, "pending": 0, "message": "no picks"}
        if values[0][:len(HEADERS)] != HEADERS:
            return {"ok": False, "settled": 0, "pending": 0, "message": "header mismatch"}

        pending_rows = []; seasons = set(); season_weeks = set(); unparseable = 0
        for row_number, row in enumerate(values[1:], start=2):
            result = row[12].strip().upper() if len(row) > 12 and row[12] else "PENDIENTE"
            if result != "PENDIENTE" or len(row) < 11:
                continue
            try:
                season = int(float(row[1])); week = int(float(row[2])); odds = float(row[6]); stake = float(row[10])
            except Exception:
                continue
            game = row[3].strip(); pick = row[4].strip()
            if " @ " not in game:
                continue
            away, home = [_normalize_team(x) for x in game.split(" @ ", 1)]
            parsed = _parse_pick(pick, home, away)
            if not parsed:
                unparseable += 1; continue
            pending_rows.append({"row_number": row_number, "season": season, "week": week,
                                 "away": away, "home": home, "parsed": parsed, "odds": odds, "stake": stake})
            seasons.add(season); season_weeks.add((season, week))

        if not pending_rows:
            return {"ok": True, "settled": 0, "pending": unparseable, "message": "no parseable pending picks"}

        try:
            schedules = nfl.import_schedules(sorted(seasons))
        except Exception:
            schedules = pd.DataFrame()

        # Cargamos el respaldo una sola vez por semana/temporada, no una vez por apuesta.
        fallback_scores = _espn_final_scores(season_weeks)
        updates = []; settled = 0; still_pending = unparseable
        source_counts = {"nflverse": 0, "espn": 0}
        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")

        for item in pending_rows:
            hs = aws = None
            source = None

            if not schedules.empty and {"week", "home_team", "away_team"}.issubset(schedules.columns):
                matches = schedules[(schedules["week"] == item["week"]) &
                                    (schedules["home_team"].map(_normalize_team) == item["home"]) &
                                    (schedules["away_team"].map(_normalize_team) == item["away"])]
                if "season" in schedules.columns:
                    matches = matches[pd.to_numeric(matches["season"], errors="coerce") == item["season"]]
                if not matches.empty:
                    gr = matches.iloc[-1]
                    nhs, naws = gr.get("home_score"), gr.get("away_score")
                    if not pd.isna(nhs) and not pd.isna(naws):
                        hs, aws = float(nhs), float(naws)
                        source = "nflverse"

            if hs is None or aws is None:
                fallback = fallback_scores.get((item["season"], item["week"], item["home"], item["away"]))
                if fallback is not None:
                    hs, aws = fallback
                    source = "espn"

            if hs is None or aws is None:
                still_pending += 1
                continue

            status = _grade(item["parsed"], item["home"], item["away"], hs, aws)
            if status is None:
                still_pending += 1
                continue
            profit = _profit(item["stake"], item["odds"], status == "GANADA", push=status == "PUSH")
            final_score = _format_final_score(item["away"], item["home"], aws, hs)
            rn = item["row_number"]
            updates.extend([
                {"range": f"{target_worksheet}!M{rn}", "majorDimension": "ROWS", "values": [[status]]},
                {"range": f"{target_worksheet}!N{rn}:O{rn}", "majorDimension": "ROWS", "values": [[profit, now_mx]]},
                {"range": f"{target_worksheet}!Q{rn}", "majorDimension": "ROWS", "values": [[final_score]]},
            ])
            settled += 1
            source_counts[source] = source_counts.get(source, 0) + 1

        if updates:
            _request_json(session, "POST", f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values:batchUpdate",
                          json={"valueInputOption": "USER_ENTERED", "data": updates})
        return {"ok": True, "settled": settled, "pending": still_pending, "worksheet": target_worksheet,
                "sources": source_counts,
                "message": "settlement complete (ML/SPREAD/TOTAL; final score saved; nflverse + ESPN final fallback)",
                "adc_project": project_id}
    except Exception as exc:
        return {"ok": False, "settled": 0, "pending": 0, "worksheet": target_worksheet,
                "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                "credential_type": type(credentials).__name__ if credentials is not None else "unresolved",
                "service_account_email": getattr(credentials, "service_account_email", None) if credentials is not None else None,
                "adc_project": project_id}