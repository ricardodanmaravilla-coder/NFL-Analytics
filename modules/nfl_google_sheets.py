from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Iterable, Mapping

SHEET_ID = os.getenv("GOOGLE_SHEETS_ID", "1VsB21QUsQL5EyXu7Sek5WVeNVznECiTuoIMMB4JXno4")
WORKSHEET = os.getenv("GOOGLE_SHEETS_WORKSHEET", "NFL_Picks")
HEADERS = [
    "Fecha", "Temporada", "Semana", "Partido", "Pick", "Probabilidad %", "Momio",
    "Edge pp", "EV %", "Kelly 1/4 %", "Apostar $", "Acción", "Resultado",
    "Profit $", "Fecha cierre", "ID",
]
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _credentials():
    import google.auth

    credentials, _ = google.auth.default(scopes=SCOPES)
    return credentials


def _record_id(season: int, week: int, row: Mapping[str, Any]) -> str:
    return f"NFL|{int(season)}|{int(week)}|{_clean(row.get('game'))}|{_clean(row.get('pick'))}"


def sync_bets(
    bets: Iterable[Mapping[str, Any]],
    season: int,
    week: int,
    bankroll: float,
    sheet_id: str | None = None,
    worksheet: str | None = None,
):
    """Insert/update NFL BET recommendations. Never raises into the scanner."""
    rows = [dict(x) for x in (bets or [])]
    if not rows:
        return {"ok": True, "inserted": 0, "updated": 0, "message": "no bets"}

    target_sheet_id = (sheet_id or SHEET_ID).strip()
    target_worksheet = (worksheet or WORKSHEET).strip() or "NFL_Picks"
    if not target_sheet_id:
        return {"ok": False, "inserted": 0, "updated": 0, "message": "sheet id missing"}

    try:
        import gspread

        client = gspread.authorize(_credentials())
        book = client.open_by_key(target_sheet_id)
        try:
            ws = book.worksheet(target_worksheet)
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(title=target_worksheet, rows=2000, cols=len(HEADERS))

        values = ws.get_all_values()
        if not values:
            ws.append_row(HEADERS, value_input_option="RAW")
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
        update_payload = []

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
                update_payload.append({"range": f"A{existing_row_number}:P{existing_row_number}", "values": [payload]})
            else:
                append_payload.append(payload)
                id_to_row[rec_id] = -1

        if update_payload:
            ws.batch_update(update_payload, value_input_option="USER_ENTERED")
        if append_payload:
            ws.append_rows(append_payload, value_input_option="USER_ENTERED")

        return {
            "ok": True,
            "inserted": len(append_payload),
            "updated": len(update_payload),
            "worksheet": target_worksheet,
            "message": "saved",
        }
    except Exception as exc:
        return {
            "ok": False,
            "inserted": 0,
            "updated": 0,
            "worksheet": target_worksheet,
            "message": f"{type(exc).__name__}: {str(exc) or repr(exc)}"[:500],
        }
