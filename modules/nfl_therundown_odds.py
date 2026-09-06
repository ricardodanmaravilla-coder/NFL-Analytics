from __future__ import annotations

"""Live NFL moneyline quotes from TheRundown V2.

Production rule: no key / no complete real quote => no executable price.
Historical nflverse lines remain for training/backtests only.
"""

import os
import time
from datetime import datetime, timezone

import requests

NFL_SPORT_ID = 2
CACHE_TTL_SECONDS = 300
_CACHE = {}
AFFILIATE_NAMES = {
    "2": "Bovada", "3": "Pinnacle", "4": "SportsBetting", "6": "BetOnline",
    "11": "LowVig", "12": "Bodog", "16": "Matchbook", "19": "DraftKings",
    "21": "Unibet", "22": "BetMGM", "23": "FanDuel", "24": "theScore Bet",
}
TEAM_ALIASES = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC",
    "LA":"LA","LAR":"LA","LV":"LV","LAC":"LAC","MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO",
    "NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB","TEN":"TEN","WAS":"WAS",
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


def _price(value):
    try:
        x = float(value)
        if abs(x) < 100:
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


def fetch_moneylines(gameday, get_fn=requests.get):
    """Return {(away,home): quote} for one NFL slate date."""
    key = os.getenv("THERUNDOWN_KEY", "").strip()
    if not key:
        return {}
    date = str(gameday)[:10]
    now = time.monotonic()
    cached = _CACHE.get(date)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return dict(cached[1])
    affiliate_ids = os.getenv("THERUNDOWN_AFFILIATE_IDS", "19,22,23")
    priority = [x.strip() for x in affiliate_ids.split(",") if x.strip()]
    url = f"https://therundown.io/api/v2/sports/{NFL_SPORT_ID}/events/{date}"
    params = {"market_ids":"1", "affiliate_ids":affiliate_ids, "main_line":"true", "hide_closed":"true", "offset":"300"}
    response = get_fn(url, params=params, headers={"X-TheRundown-Key":key, "Accept":"application/json"}, timeout=12)
    if getattr(response, "status_code", 0) != 200:
        return {}
    payload = response.json()
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
                    prices = line.get("prices") or {}
                    if not isinstance(prices, dict):
                        continue
                    for aid in priority:
                        if aid not in prices:
                            continue
                        pobj = _latest_price_obj(prices[aid])
                        if pobj.get("is_main_line") is False:
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
    _CACHE[date] = (now, out)
    return dict(out)


def get_moneyline(home, away, gameday, get_fn=requests.get):
    quotes = fetch_moneylines(gameday, get_fn=get_fn)
    return quotes.get((_norm_team(away), _norm_team(home)))
