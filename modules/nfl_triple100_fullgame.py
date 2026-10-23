"""Triple-80 full-game scanner using nflverse schedule data (no Linemate scraping).

Markets are evaluated against today's candidate line, not old closing lines.
A strict push does NOT count as a win. Samples can overlap by design.
"""
from __future__ import annotations
import math
import pandas as pd

MIN_SAMPLE = 4
MIN_RATE = 0.80
WINDOWS = {"general": 11, "h2h": 8, "venue": 8}
SUPPORTED = {"moneyline", "spread", "total_over", "total_under"}


def load_nflverse_schedules(seasons=None):
    import nfl_data_py as nfl
    if seasons is None:
        from datetime import date
        y = date.today().year
        seasons = list(range(1999, y + 1))
    return nfl.import_schedules([int(y) for y in seasons])


def _history(schedules, kickoff):
    df = schedules.copy()
    required = {"home_team", "away_team", "home_score", "away_score", "gameday"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing schedule columns: {sorted(missing)}")
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    before = pd.to_datetime(kickoff)
    if pd.isna(before):
        raise ValueError("invalid kickoff")
    df = df[(df["gameday"] < before) & df["home_score"].notna()
            & df["away_score"].notna()].copy()
    # Dates in nflverse have day granularity. Exclude same-day games to
    # prevent using results unavailable when a pick was made.
    if "game_type" in df:
        df = df[df["game_type"].isin(("REG", "WC", "DIV", "CON", "SB", "POST"))]
    return df.sort_values("gameday", ascending=False)


def evaluate(schedules, *, team, opponent, kickoff, venue, market, line=None,
             min_games=4, windows=None):
    """Return diagnostic for one candidate selection.

    venue: 'home' or 'away' for the selection team.
    spread line from selection team's perspective, e.g. +3.5.
    total over/under tests combined final points from games involving team.
    """
    if min_games < MIN_SAMPLE:
        raise ValueError("min_games cannot be below 4")
    if market not in SUPPORTED or venue not in ("home", "away"):
        raise ValueError("invalid market or venue")
    if not team or not opponent or team == opponent:
        raise ValueError("invalid teams")
    w = WINDOWS if windows is None else windows
    if any(int(w[k]) < min_games for k in WINDOWS):
        raise ValueError("every window must be >= min_games")
    if market != "moneyline" and (line is None or not math.isfinite(float(line))):
        raise ValueError("finite line required")
    history = _history(schedules, kickoff)
    involve = history[(history.home_team == team) | (history.away_team == team)]
    h2h = involve[((involve.home_team == opponent) | (involve.away_team == opponent))]
    venue_rows = involve[involve.home_team == team] if venue == "home" else involve[involve.away_team == team]
    pools = {"general": involve, "h2h": h2h, "venue": venue_rows}
    out = {}
    for name, pool in pools.items():
        pool = pool.head(int(w[name]))
        wins = 0
        pushes = 0
        for _, row in pool.iterrows():
            selected_score = float(row.home_score if row.home_team == team else row.away_score)
            other_score = float(row.away_score if row.home_team == team else row.home_score)
            margin = selected_score - other_score
            if market == "moneyline":
                score = margin
            elif market == "spread":
                score = margin + float(line)
            elif market == "total_over":
                score = selected_score + other_score - float(line)
            else:
                score = float(line) - selected_score - other_score
            if score > 0:
                wins += 1
            elif score == 0:
                pushes += 1
        count = len(pool)
        good = count >= min_games and (wins / count) >= MIN_RATE
        out[name] = {"wins": wins, "games": count, "pushes": pushes,
                     "rate": wins / count if count else None, "passed": good}
    return {"team": team, "opponent": opponent, "kickoff": str(kickoff),
            "venue": venue, "market": market, "line": line,
            "triple_80": all(v["passed"] for v in out.values()),
            "triple_100": all(v["passed"] for v in out.values()),  # legacy field: compatibility
            "minimum_rate": MIN_RATE,
            "samples": out, "source": "nflverse_schedules"}


def scan(schedules, candidates):
    return [evaluate(schedules, **row) for row in candidates]
