"""Live full-game Triple 80 scan using TheRundown market snapshots.

Never manufacture prices: missing market legs are omitted. Data source nflverse
for results, TheRundown for actionable candidate lines.
"""
from .nfl_therundown_odds import get_markets, configured
from .nfl_triple100_fullgame import evaluate

def scan_live(schedules, upcoming):
    """upcoming: iterable of {home_team, away_team, gameday} (unplayed games).
    Returns approved trend rows plus diagnostics. No bets or sheet writes.
    """
    if not configured():
        return {"ok": False, "reason": "THERUNDOWN_KEYS_NOT_CONFIGURED",
                "approved": [], "checked": 0, "unpriced_games": []}
    approved = []
    unpriced = []
    checked = 0
    for game in upcoming:
        home, away, day = game["home_team"], game["away_team"], str(game["gameday"])[:10]
        quote = get_markets(home, away, day)
        if not quote:
            unpriced.append({"home_team": home, "away_team": away, "gameday": day,
                             "reason": "NO_LIVE_QUOTES"})
            continue
        rows = []
        if quote.get("home_moneyline") is not None and quote.get("away_moneyline") is not None:
            rows.extend([("moneyline",home,"home",None,quote["home_moneyline"],quote.get("book")),
                         ("moneyline",away,"away",None,quote["away_moneyline"],quote.get("book"))])
        if all(quote.get(x) is not None for x in ("home_spread","away_spread","home_spread_odds","away_spread_odds")):
            rows.extend([("spread",home,"home",quote["home_spread"],quote["home_spread_odds"],quote.get("spread_book")),
                         ("spread",away,"away",quote["away_spread"],quote["away_spread_odds"],quote.get("spread_book"))])
        if all(quote.get(x) is not None for x in ("total_line","over_odds","under_odds")):
            rows.extend([("total_over",home,"home",quote["total_line"],quote["over_odds"],quote.get("total_book")),
                         ("total_under",home,"home",quote["total_line"],quote["under_odds"],quote.get("total_book"))])
        if not rows:
            unpriced.append({"home_team":home,"away_team":away,"gameday":day,
                             "reason":"INCOMPLETE_MARKETS"})
        for market, team, venue, line, odd, book in rows:
            opponent = away if team == home else home
            result = evaluate(schedules, team=team, opponent=opponent,
                              kickoff=day, venue=venue, market=market, line=line)
            checked += 1
            if result["triple_80"]:
                result.update({"american_odds":odd,"book":book,
                               "fetched_at":quote.get("fetched_at"),
                               "odds_source":"TheRundown",
                               "slate_date":quote.get("slate_date")})
                approved.append(result)
    return {"ok": True, "approved": approved, "checked": checked,
            "unpriced_games": unpriced}
