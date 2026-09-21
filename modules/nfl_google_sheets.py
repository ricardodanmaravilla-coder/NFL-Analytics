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
_ESPN_TEAM_ALIASES = {"WSH": "WAS", "JAC": "JAX", "LAR": "LA"}
_SPREAD_RE = re.compile(r"^(.+?)\s+([+-][0-9]+(?:\.[0-9]+)?)$")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).replace("\n", " ").strip()


def _normalize_team(team: Any) -> str:
    value = _clean(team).upper()
    return _ESPN_TEAM_ALIASES.get(value, value)


def _credentials():
    import google.auth
    return google.auth.default(scopes=SCOPES)


def _record_id(season: int, week: int, row: Mapping[str, Any]) -> str:
    return f"NFL|{int(season)}|{int(week)}|{_clean(row.get('game'))}|{_clean(row.get('pick'))}"


def _request_json(session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    if not response.ok:
        raise RuntimeError(f"Sheets API {response.status_code}: {response.text[:1000]}")
    return response.json() if response.content else {}


def _total_key(game: Any, pick: Any):
    """Identity of a total ignoring line movement: (game, OVER/UNDER)."""
    p = _clean(pick)
    m = re.fullmatch(r"(Over|Under)\s+[0-9]+(?:\.[0-9]+)?", p, flags=re.I)
    if not m:
        return None
    return (_clean(game), m.group(1).upper())


def _spread_key(game: Any, pick: Any):
    """Identidad de spread ignorando la línea: (partido, equipo).

    DEN +2.5, DEN +3 y DEN +3.5 producen la misma llave. ML y Totals devuelven None.
    """
    match = _SPREAD_RE.fullmatch(_clean(pick))
    if not match:
        return None
    return (_clean(game), _normalize_team(match.group(1)))


def _profit(stake: float, odds: float, won: bool, push: bool = False) -> float:
    if push:
        return 0.0
    if not won:
        return round(-abs(float(stake)), 2)
    stake, odds = abs(float(stake)), float(odds)
    if odds > 0:
        return round(stake * odds / 100.0, 2)
    if odds < 0:
        return round(stake * 100.0 / abs(odds), 2)
    return 0.0


_profit_for_moneyline = _profit


def _format_final_score(away: str, home: str, away_score: float, home_score: float) -> str:
    def fmt(v):
        n = float(v)
        return str(int(n)) if n.is_integer() else str(n)
    return f"{away} {fmt(away_score)} - {home} {fmt(home_score)}"


def sync_bets(bets: Iterable[Mapping[str, Any]], season: int, week: int, bankroll: float,
              sheet_id: str | None = None, worksheet: str | None = None):
    """Inserta BET inmutables y bloquea variantes posteriores del mismo spread/equipo.

    La primera línea de spread registrada para un equipo en ese partido/semana queda fija.
    Ejemplo: si existe DEN +3, DEN +2.5 o DEN +3.5 se omiten. Esto NO afecta ML ni Totals.
    """
    rows = [dict(x) for x in (bets or [])]
    if not rows:
        return {"ok": True, "inserted": 0, "updated": 0, "skipped_existing": 0,
                "skipped_spread_variant": 0, "message": "no bets"}

    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    if not target_sheet_id:
        return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0,
                "skipped_spread_variant": 0, "message": "sheet id missing"}

    credentials = None; project_id = None
    try:
        from google.auth.transport.requests import AuthorizedSession
        credentials, project_id = _credentials()
        session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded = quote(f"{target_worksheet}!A:Q", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded}").get("values", [])

        if not values:
            hr = quote(f"{target_worksheet}!A1:Q1", safe="")
            _request_json(session, "PUT", f"{base}/{hr}?valueInputOption=RAW",
                          json={"range": f"{target_worksheet}!A1:Q1", "majorDimension": "ROWS", "values": [HEADERS]})
            values = [HEADERS]
        elif values[0][:len(HEADERS)] != HEADERS:
            return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0,
                    "skipped_spread_variant": 0, "worksheet": target_worksheet,
                    "message": "header mismatch; existing sheet preserved"}

        existing_ids = {r[15] for r in values[1:] if len(r) >= 16 and r[15]}
        existing_spreads = set(); existing_totals = set()
        for r in values[1:]:
            if len(r) < 5:
                continue
            try:
                if int(float(r[1])) != int(season) or int(float(r[2])) != int(week):
                    continue
            except Exception:
                continue
            key = _spread_key(r[3], r[4])
            if key is not None:
                existing_spreads.add(key)
            total_key = _total_key(r[3], r[4])
            if total_key is not None:
                existing_totals.add(total_key)

        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
        payload, seen_ids, seen_spreads, seen_totals = [], set(), set(), set()
        skipped = skipped_spread_variant = skipped_total_variant = 0
        for bet in rows:
            rec_id = _record_id(season, week, bet)
            if rec_id in existing_ids or rec_id in seen_ids:
                skipped += 1
                continue

            key = _spread_key(bet.get("game"), bet.get("pick"))
            if key is not None and (key in existing_spreads or key in seen_spreads):
                skipped += 1
                skipped_spread_variant += 1
                continue

            total_key = _total_key(bet.get("game"), bet.get("pick"))
            if total_key is not None and (total_key in existing_totals or total_key in seen_totals):
                skipped += 1
                skipped_total_variant += 1
                continue

            seen_ids.add(rec_id)
            if key is not None:
                seen_spreads.add(key)
            if total_key is not None:
                seen_totals.add(total_key)
            payload.append([
                now_mx, int(season), int(week), _clean(bet.get("game")), _clean(bet.get("pick")),
                bet.get("probability", ""), bet.get("odds", ""), bet.get("edge", ""), bet.get("ev", ""),
                bet.get("kelly", ""), bet.get("stake", ""), "BET", "PENDIENTE", "", "", rec_id, "",
            ])

        if payload:
            ar = quote(f"{target_worksheet}!A:Q", safe="")
            _request_json(session, "POST", f"{base}/{ar}:append?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS",
                          json={"majorDimension": "ROWS", "values": payload})

        return {"ok": True, "inserted": len(payload), "updated": 0, "skipped_existing": skipped,
                "skipped_spread_variant": skipped_spread_variant, "skipped_total_variant": skipped_total_variant, "worksheet": target_worksheet,
                "message": "saved via Sheets API; first spread and total side per game preserved; ML unaffected",
                "credential_type": type(credentials).__name__,
                "service_account_email": getattr(credentials, "service_account_email", None), "adc_project": project_id}
    except Exception as exc:
        return {"ok": False, "inserted": 0, "updated": 0, "skipped_existing": 0,
                "skipped_spread_variant": 0, "worksheet": target_worksheet,
                "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                "credential_type": type(credentials).__name__ if credentials is not None else "unresolved",
                "service_account_email": getattr(credentials, "service_account_email", None) if credentials is not None else None,
                "adc_project": project_id}


def _parse_pick(pick: str, home: str, away: str):
    p = _clean(pick)
    home, away = _normalize_team(home), _normalize_team(away)
    if p.upper().endswith(" ML"):
        team = _normalize_team(p[:-3])
        if team in {home, away}:
            return {"market": "ML", "team": team}
    m = re.fullmatch(r"(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", p, flags=re.I)
    if m:
        return {"market": "TOTAL", "side": m.group(1).upper(), "line": float(m.group(2))}
    m = _SPREAD_RE.fullmatch(p)
    if m:
        team = _normalize_team(m.group(1))
        if team in {home, away}:
            return {"market": "SPREAD", "team": team, "line": float(m.group(2))}
    return None


def _grade(parsed, home, away, home_score, away_score):
    home, away = _normalize_team(home), _normalize_team(away)
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


def _espn_final_scores(season_week_pairs):
    import requests
    out = {}
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    headers = {"User-Agent": "NFL-Analytics/3.9 settlement"}
    for season, nfl_week in sorted(set(season_week_pairs)):
        season_type = 2 if int(nfl_week) <= 18 else 3
        espn_week = int(nfl_week) if season_type == 2 else max(1, int(nfl_week) - 18)
        try:
            response = requests.get(url, params={"dates": int(season), "seasontype": season_type,
                                    "week": espn_week, "limit": 100}, headers=headers, timeout=12)
            response.raise_for_status(); payload = response.json()
        except Exception:
            continue
        for event in payload.get("events", []) or []:
            if not bool((((event.get("status") or {}).get("type") or {}).get("completed"))):
                continue
            comps = event.get("competitions") or []
            if not comps:
                continue
            home = away = None; hs = aws = None
            for c in comps[0].get("competitors") or []:
                team = _normalize_team((c.get("team") or {}).get("abbreviation"))
                try: score = float(c.get("score"))
                except Exception: score = None
                if c.get("homeAway") == "home": home, hs = team, score
                elif c.get("homeAway") == "away": away, aws = team, score
            if home and away and hs is not None and aws is not None:
                out[(int(season), int(nfl_week), home, away)] = (hs, aws)
    return out


def settle_pending(sheet_id: str | None = None, worksheet: str | None = None):
    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    credentials = None; project_id = None
    try:
        import pandas as pd
        import nfl_data_py as nfl
        from google.auth.transport.requests import AuthorizedSession
        credentials, project_id = _credentials(); session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded = quote(f"{target_worksheet}!A:Q", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded}").get("values", [])
        if len(values) <= 1:
            return {"ok": True, "settled": 0, "pending": 0, "message": "no picks"}
        if values[0][:len(HEADERS)] != HEADERS:
            return {"ok": False, "settled": 0, "pending": 0, "message": "header mismatch"}

        pending_rows, seasons, season_weeks = [], set(), set(); unparseable = 0
        for rn, row in enumerate(values[1:], start=2):
            result = row[12].strip().upper() if len(row) > 12 and row[12] else "PENDIENTE"
            if result != "PENDIENTE" or len(row) < 11: continue
            try:
                season = int(float(row[1])); week = int(float(row[2])); odds = float(row[6]); stake = float(row[10])
            except Exception: continue
            game, pick = row[3].strip(), row[4].strip()
            if " @ " not in game: continue
            away, home = [_normalize_team(x) for x in game.split(" @ ", 1)]
            parsed = _parse_pick(pick, home, away)
            if not parsed:
                unparseable += 1; continue
            pending_rows.append({"row_number": rn, "season": season, "week": week, "away": away,
                                 "home": home, "parsed": parsed, "odds": odds, "stake": stake})
            seasons.add(season); season_weeks.add((season, week))

        if not pending_rows:
            return {"ok": True, "settled": 0, "pending": unparseable, "message": "no parseable pending picks"}
        try: schedules = nfl.import_schedules(sorted(seasons))
        except Exception: schedules = pd.DataFrame()
        fallback_scores = _espn_final_scores(season_weeks)
        updates = []; settled = 0; still_pending = unparseable
        source_counts = {"nflverse": 0, "espn": 0}
        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")

        for item in pending_rows:
            hs = aws = source = None
            if not schedules.empty and {"week", "home_team", "away_team"}.issubset(schedules.columns):
                matches = schedules[(schedules["week"] == item["week"]) &
                                    (schedules["home_team"].map(_normalize_team) == item["home"]) &
                                    (schedules["away_team"].map(_normalize_team) == item["away"])]
                if "season" in schedules.columns:
                    matches = matches[pd.to_numeric(matches["season"], errors="coerce") == item["season"]]
                if not matches.empty:
                    gr = matches.iloc[-1]; nhs, naws = gr.get("home_score"), gr.get("away_score")
                    if not pd.isna(nhs) and not pd.isna(naws): hs, aws, source = float(nhs), float(naws), "nflverse"
            if hs is None or aws is None:
                fallback = fallback_scores.get((item["season"], item["week"], item["home"], item["away"]))
                if fallback is not None: hs, aws, source = fallback[0], fallback[1], "espn"
            if hs is None or aws is None:
                still_pending += 1; continue
            status = _grade(item["parsed"], item["home"], item["away"], hs, aws)
            if status is None:
                still_pending += 1; continue
            profit = _profit(item["stake"], item["odds"], status == "GANADA", push=status == "PUSH")
            score = _format_final_score(item["away"], item["home"], aws, hs); rn = item["row_number"]
            updates.extend([
                {"range": f"{target_worksheet}!M{rn}", "majorDimension": "ROWS", "values": [[status]]},
                {"range": f"{target_worksheet}!N{rn}:O{rn}", "majorDimension": "ROWS", "values": [[profit, now_mx]]},
                {"range": f"{target_worksheet}!Q{rn}", "majorDimension": "ROWS", "values": [[score]]},
            ])
            settled += 1; source_counts[source] = source_counts.get(source, 0) + 1
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
