from __future__ import annotations

import os
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
        body = response.text[:1000]
        raise RuntimeError(f"Sheets API {response.status_code}: {body}")
    if not response.content:
        return {}
    return response.json()


def sync_bets(
    bets: Iterable[Mapping[str, Any]],
    season: int,
    week: int,
    bankroll: float,
    sheet_id: str | None = None,
    worksheet: str | None = None,
):
    """Insert/update NFL BET recommendations using the Google Sheets REST API."""
    rows = [dict(x) for x in (bets or [])]
    if not rows:
        return {"ok": True, "inserted": 0, "updated": 0, "message": "no bets"}

    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    if not target_sheet_id:
        return {"ok": False, "inserted": 0, "updated": 0, "message": "sheet id missing"}

    credentials = None
    project_id = None
    try:
        from google.auth.transport.requests import AuthorizedSession

        credentials, project_id = _credentials()
        session = AuthorizedSession(credentials)
        base = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values"
        encoded_range = quote(f"{target_worksheet}!A:P", safe="")
        values_resp = _request_json(session, "GET", f"{base}/{encoded_range}")
        values = values_resp.get("values", [])

        if not values:
            header_range = quote(f"{target_worksheet}!A1:P1", safe="")
            _request_json(
                session,
                "PUT",
                f"{base}/{header_range}?valueInputOption=RAW",
                json={"range": f"{target_worksheet}!A1:P1", "majorDimension": "ROWS", "values": [HEADERS]},
            )
            values = [HEADERS]
        elif values[0][: len(HEADERS)] != HEADERS:
            return {
                "ok": False,
                "inserted": 0,
                "updated": 0,
                "worksheet": target_worksheet,
                "message": "header mismatch; existing sheet preserved",
            }

        id_to_row = {}
        for idx, existing in enumerate(values[1:], start=2):
            if len(existing) >= 16 and existing[15]:
                id_to_row[existing[15]] = idx

        now_mx = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
        append_payload = []
        updates = []

        for bet in rows:
            rec_id = _record_id(season, week, bet)
            existing_row_number = id_to_row.get(rec_id)
            previous = values[existing_row_number - 1] if existing_row_number else []
            previous_result = previous[12] if len(previous) > 12 and previous[12] else "PENDIENTE"
            previous_profit = previous[13] if len(previous) > 13 else ""
            previous_close = previous[14] if len(previous) > 14 else ""

            payload = [
                now_mx,
                int(season),
                int(week),
                _clean(bet.get("game")),
                _clean(bet.get("pick")),
                bet.get("probability", ""),
                bet.get("odds", ""),
                bet.get("edge", ""),
                bet.get("ev", ""),
                bet.get("kelly", ""),
                bet.get("stake", ""),
                "BET",
                previous_result,
                previous_profit,
                previous_close,
                rec_id,
            ]

            if existing_row_number:
                updates.append({
                    "range": f"{target_worksheet}!A{existing_row_number}:P{existing_row_number}",
                    "majorDimension": "ROWS",
                    "values": [payload],
                })
            else:
                append_payload.append(payload)

        if updates:
            batch_url = f"https://sheets.googleapis.com/v4/spreadsheets/{target_sheet_id}/values:batchUpdate"
            _request_json(
                session,
                "POST",
                batch_url,
                json={"valueInputOption": "USER_ENTERED", "data": updates},
            )

        if append_payload:
            append_range = quote(f"{target_worksheet}!A:P", safe="")
            _request_json(
                session,
                "POST",
                f"{base}/{append_range}:append?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS",
                json={"majorDimension": "ROWS", "values": append_payload},
            )

        service_account_email = getattr(credentials, "service_account_email", None)
        return {
            "ok": True,
            "inserted": len(append_payload),
            "updated": len(updates),
            "worksheet": target_worksheet,
            "message": "saved via Sheets API",
            "credential_type": type(credentials).__name__,
            "service_account_email": service_account_email,
            "adc_project": project_id,
        }
    except Exception as exc:
        return {
            "ok": False,
            "inserted": 0,
            "updated": 0,
            "worksheet": target_worksheet,
            "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:1000],
            "credential_type": type(credentials).__name__ if credentials is not None else "unresolved",
            "service_account_email": getattr(credentials, "service_account_email", None) if credentials is not None else None,
            "adc_project": project_id,
        }
