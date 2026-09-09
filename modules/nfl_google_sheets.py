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
    "Profit $", "Fecha cierre", "ID",
]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


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
        encoded_range = quote(f"{target_worksheet}!A:P", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded_range}").get("values", [])

        if not values:
            header_range = quote(f"{target_worksheet}!A1:P1", safe="")
            _request_json(session, "PUT", f"{base}/{header_range}?valueInputOption=RAW",
                          json={"range": f"{target_worksheet}!A1:P1", "majorDimension": "ROWS", "values": [HEADERS]})
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
                bet.get("kelly", ""), bet.get("stake", ""), "BET", "PENDIENTE", "", "", rec_id,
            ])

        if payload:
            append_range = quote(f"{target_worksheet}!A:P", safe="")
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


def settle_pending(sheet_id: str | None = None, worksheet: str | None = None):
    """Liquida BET pendientes de Moneyline, spread y total usando marcador final real."""
    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    credentials = None; project_id = None
    try:
        import pandas as pd
        import nfl_data_py as nfl
        from google.auth.transport.requests import AuthorizedSession

        credentials, project_id = _credentials(); session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded_range = quote(f"{target_worksheet}!A:P", safe="")
        values = _request_json(session, "GET", f"{base}/{encoded_range}").get("values", [])
        if len(values) <= 1:
            return {"ok": True, "settled": 0, "pending": 0, "message": "no picks"}
        if values[0][:len(HEADERS)] != HEADERS:
            return {"ok": False, "settled": 0, "pending": 0, "message": "header mismatch"}

        pending_rows = []; seasons = set(); unparseable = 0
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
            away, home = [x.strip() for x in game.split(" @ ", 1)]
            parsed = _parse_pick(pick, home, away)
            if not parsed:
                unparseable += 1; continue
            pending_rows.append({"row_number": row_number, "season": season, "week": week,
                                 "away": away, "home": home, "parsed": parsed, "odds": odds, "stake": stake})
            seasons.add(season)

        if not pending_rows:
            return {"ok": True, "settled": 0, "pending": unparseable, "message": "no parseable pending picks"}

        schedules = nfl.import_schedules(sorted(seasons))
        updates = []; settled = 0; still_pending = unparseable
        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
        for item in pending_rows:
            matches = schedules[(schedules["week"] == item["week"]) &
                                (schedules["home_team"] == item["home"]) &
                                (schedules["away_team"] == item["away"])]
            if "season" in schedules.columns:
                matches = matches[schedules.loc[matches.index, "season"] == item["season"]]
            if matches.empty:
                still_pending += 1; continue
            gr = matches.iloc[-1]; hs = gr.get("home_score"); aws = gr.get("away_score")
            if pd.isna(hs) or pd.isna(aws):
                still_pending += 1; continue
            status = _grade(item["parsed"], item["home"], item["away"], float(hs), float(aws))
            if status is None:
                still_pending += 1; continue
            profit = _profit(item["stake"], item["odds"], status == "GANADA", push=status == "PUSH")
            rn = item["row_number"]
            updates.extend([
                {"range": f"{target_worksheet}!M{rn}", "majorDimension": "ROWS", "values": [[status]]},
                {"range": f"{target_worksheet}!N{rn}:O{rn}", "majorDimension": "ROWS", "values": [[profit, now_mx]]},
            ])
            settled += 1

        if updates:
            _request_json(session, "POST", f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values:batchUpdate",
                          json={"valueInputOption": "USER_ENTERED", "data": updates})
        return {"ok": True, "settled": settled, "pending": still_pending, "worksheet": target_worksheet,
                "message": "settlement complete (ML/SPREAD/TOTAL)", "adc_project": project_id}
    except Exception as exc:
        return {"ok": False, "settled": 0, "pending": 0, "worksheet": target_worksheet,
                "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
                "credential_type": type(credentials).__name__ if credentials is not None else "unresolved",
                "service_account_email": getattr(credentials, "service_account_email", None) if credentials is not None else None,
                "adc_project": project_id}
