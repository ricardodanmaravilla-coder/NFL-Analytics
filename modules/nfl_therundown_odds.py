from __future__ import annotations

"""Live NFL moneyline quotes from TheRundown V2.

Production rule: no key / no complete real quote => no executable price.
Historical nflverse lines remain for training/backtests only.
"""

import os
import time
from datetime import date as date_type, datetime, timedelta, timezone

import requests

NFL_SPORT_ID = 2
CACHE_TTL_SECONDS = 300
_CACHE = {}
DEFAULT_AFFILIATE_IDS = "19,22,23,3,2,6,4,11,12,21,24"
AFFILIATE_NAMES = {
    "2": "Bovada", "3": "Pinnacle", "4": "SportsBetting", "6": "BetOnline",
    "11": "LowVig", "12": "Bodog", "16": "Matchbook", "19": "DraftKings",
    "21": "Unibet", "22": "BetMGM", "23": "FanDuel", "24": "theScore Bet",
}
TEAM_ALIASES = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","GNB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","JAC":"JAX","KC":"KC","KAN":"KC",
    "LA":"LA","LAR":"LA","LV":"LV","LVR":"LV","LAC":"LAC","MIA":"MIA","MIN":"MIN","NE":"NE","NWE":"NE","NO":"NO","NOR":"NO",
    "NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","SFO":"SF","TB":"TB","TAM":"TB","TEN":"TEN",
    "WAS":"WAS","WSH":"WAS","OAK":"LV","SD":"LAC","STL":"LA",
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LOS ANGELES RAMS":"LA","LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SEATTLE SEAHAWKS":"SEA",
    "SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
}


def _norm_team(value):
    key = " ".join(str(value or "").upper().replace(".", "").split())
    return TEAM_ALIASES.get(key, key)


def configured():
    return bool(os.getenv("THERUNDOWN_KEY", "").strip())


def _priority():
    affiliate_ids = os.getenv("THERUNDOWN_AFFILIATE_IDS", DEFAULT_AFFILIATE_IDS)
    return [x.strip() for x in affiliate_ids.split(",") if x.strip()]


def _price(value):
    try:
        x = float(value)
        if x == 0.0001 or abs(x) < 100:
            return None
        return int(round(x))
    except (TypeError, ValueError):
        return None


def _latest_price_obj(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        return {"price": value}
    rows = [r for r in value if isinstance(r, dict)]
    if not rows:
        return {}
    def stamp(row):
        return str(row.get("updated_at") or row.get("last_updated") or row.get("created_at") or "")
    return sorted(rows, key=stamp)[-1]


def _team_pair(event):
    normalized = [x for x in (event.get("teams_normalized") or []) if isinstance(x, dict)]
    if len(normalized) >= 2:
        away = next((x for x in normalized if x.get("is_home") is False or x.get("is_away") is True), normalized[0])
        home = next((x for x in normalized if x.get("is_home") is True), None)
        if home is None:
            home = next((x for x in normalized if x is not away), normalized[1])
        return _norm_team(away.get("abbreviation") or away.get("name")), _norm_team(home.get("abbreviation") or home.get("name"))
    teams = [x for x in (event.get("teams") or []) if isinstance(x, dict)]
    if len(teams) < 2:
        return None, None
    away = next((x for x in teams if x.get("is_away") is True), teams[0])
    home = next((x for x in teams if x.get("is_home") is True), None)
    if home is None:
        home = next((x for x in teams if x is not away), teams[1])
    return _norm_team(away.get("abbreviation") or away.get("name")), _norm_team(home.get("abbreviation") or home.get("name"))


def _request_snapshot(gameday, get_fn=requests.get):
    key = os.getenv("THERUNDOWN_KEY", "").strip()
    if not key:
        return None, _priority(), str(gameday)[:10]
    date = str(gameday)[:10]
    priority = _priority()
    affiliate_ids = ",".join(priority)
    url = f"https://therundown.io/api/v2/sports/{NFL_SPORT_ID}/events/{date}"
    params = {
        "market_ids": "1",
        "affiliate_ids": affiliate_ids,
        "main_line": "true",
        "hide_closed": "true",
        "offset": "300",
    }
    response = get_fn(url, params=params, headers={"X-TheRundown-Key": key, "Accept": "application/json"}, timeout=12)
    return response, priority, date


def _extract_quotes(payload, priority):
    events = payload.get("events", []) if isinstance(payload, dict) else []
    out = {}
    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for event in events:
        if not isinstance(event, dict):
            continue
        away, home = _team_pair(event)
        if not away or not home:
            continue
        by_book = {aid: {} for aid in priority}
        for market in event.get("markets") or []:
            if not isinstance(market, dict) or int(market.get("market_id") or 0) != 1:
                continue
            if market.get("period_id") not in (None, "", 0, "0"):
                continue
            for participant in market.get("participants") or []:
                if not isinstance(participant, dict):
                    continue
                sel = _norm_team(participant.get("abbreviation") or participant.get("name") or participant.get("team_name"))
                if sel not in {home, away}:
                    continue
                lines = participant.get("lines") or []
                if isinstance(lines, dict):
                    lines = [lines]
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    prices = line.get("prices") or {}
                    if not isinstance(prices, dict):
                        continue
                    for aid in priority:
                        if aid not in prices:
                            continue
                        pobj = _latest_price_obj(prices[aid])
                        if pobj.get("is_main_line") is False or pobj.get("closed_at") not in (None, ""):
                            continue
                        p = _price(pobj.get("price", pobj.get("odds")))
                        if p is not None:
                            by_book[aid][sel] = p
        chosen = next((aid for aid in priority if home in by_book[aid] and away in by_book[aid]), None)
        if chosen:
            out[(away, home)] = {
                "home_moneyline": by_book[chosen][home],
                "away_moneyline": by_book[chosen][away],
                "book": AFFILIATE_NAMES.get(chosen, f"TheRundown {chosen}"),
                "affiliate_id": chosen,
                "source": "TheRundown",
                "fetched_at": fetched_at,
            }
    return out


def fetch_moneylines(gameday, get_fn=requests.get):
    if not configured():
        return {}
    date = str(gameday)[:10]
    now = time.monotonic()
    cached = _CACHE.get(date)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return dict(cached[1])
    response, priority, _ = _request_snapshot(gameday, get_fn=get_fn)
    if response is None or getattr(response, "status_code", 0) != 200:
        return {}
    try:
        payload = response.json()
    except Exception:
        return {}
    out = _extract_quotes(payload, priority)
    _CACHE[date] = (now, out)
    return dict(out)


def _candidate_slate_dates(gameday):
    """Search nearby TheRundown snapshots while keeping matchup identity strict.

    Exact kickoff date is tried first, then six days backward (weekly slate grouping),
    then two days forward for feeds that file an event under a later slate snapshot.
    No quote is accepted unless normalized away/home match exactly.
    """
    raw = str(gameday)[:10]
    try:
        base = date_type.fromisoformat(raw)
    except ValueError:
        return [raw]
    offsets = [0, -1, -2, -3, -4, -5, -6, 1, 2]
    return [(base + timedelta(days=days)).isoformat() for days in offsets]


def diagnose_date(gameday, get_fn=requests.get):
    date = str(gameday)[:10]
    result = {"configured": configured(), "sport_id": NFL_SPORT_ID, "date": date, "market_id": 1, "affiliate_priority": _priority()}
    if not configured():
        result.update({"ok": False, "reason": "THERUNDOWN_KEY_NOT_CONFIGURED"})
        return result
    try:
        response, priority, _ = _request_snapshot(gameday, get_fn=get_fn)
        status = int(getattr(response, "status_code", 0) or 0)
        result["http_status"] = status
        headers = getattr(response, "headers", {}) or {}
        result["datapoints"] = headers.get("X-Datapoints") or headers.get("x-datapoints")
        if status != 200:
            result.update({"ok": False, "reason": f"HTTP_{status}"})
            return result
        payload = response.json()
        events = payload.get("events", []) if isinstance(payload, dict) else []
        offered = set(); moneyline_markets = 0; event_rows = []
        for event in events:
            if not isinstance(event, dict):
                continue
            away, home = _team_pair(event); books_for_event = set()
            for market in event.get("markets") or []:
                if not isinstance(market, dict) or int(market.get("market_id") or 0) != 1:
                    continue
                if market.get("period_id") not in (None, "", 0, "0"):
                    continue
                moneyline_markets += 1
                for participant in market.get("participants") or []:
                    for line in participant.get("lines") or []:
                        if not isinstance(line, dict):
                            continue
                        prices = line.get("prices") or {}
                        if isinstance(prices, dict):
                            books_for_event.update(str(x) for x in prices.keys())
            offered.update(books_for_event)
            event_rows.append({"game": f"{away} @ {home}" if away and home else str(event.get("event_id") or "unknown"), "affiliate_ids": sorted(books_for_event)})
        quotes = _extract_quotes(payload, priority)
        result.update({"ok": True, "events": len(events), "moneyline_markets": moneyline_markets, "complete_moneyline_quotes": len(quotes), "offered_affiliate_ids": sorted(offered), "offered_books": [AFFILIATE_NAMES.get(aid, f"TheRundown {aid}") for aid in sorted(offered)], "games": event_rows})
        if not events: result["reason"] = "NO_EVENTS_FOR_DATE"
        elif moneyline_markets == 0: result["reason"] = "NO_MONEYLINE_MARKET"
        elif not quotes: result["reason"] = "NO_COMPLETE_TWO_SIDED_MONEYLINE"
        else: result["reason"] = "OK"
        return result
    except Exception as exc:
        result.update({"ok": False, "reason": f"{type(exc).__name__}"})
        return result


def get_moneyline(home, away, gameday, get_fn=requests.get):
    target = (_norm_team(away), _norm_team(home))
    for slate_date in _candidate_slate_dates(gameday):
        quote = fetch_moneylines(slate_date, get_fn=get_fn).get(target)
        if quote:
            out = dict(quote)
            out["slate_date"] = slate_date
            return out
    return None
