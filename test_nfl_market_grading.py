from modules.nfl_google_sheets import _grade, _parse_pick


def test_parse_and_grade_moneyline():
    p = _parse_pick("SEA ML", "SEA", "NE")
    assert p == {"market": "ML", "team": "SEA"}
    assert _grade(p, "SEA", "NE", 24, 17) == "GANADA"


def test_parse_and_grade_home_spread():
    p = _parse_pick("SEA -3.5", "SEA", "NE")
    assert p["market"] == "SPREAD" and p["line"] == -3.5
    assert _grade(p, "SEA", "NE", 24, 20) == "GANADA"
    assert _grade(p, "SEA", "NE", 23, 20) == "PERDIDA"


def test_parse_and_grade_away_spread_and_push():
    p = _parse_pick("NE +3", "SEA", "NE")
    assert _grade(p, "SEA", "NE", 24, 21) == "PUSH"
    assert _grade(p, "SEA", "NE", 24, 22) == "GANADA"


def test_parse_and_grade_totals():
    over = _parse_pick("Over 44.5", "SEA", "NE")
    under = _parse_pick("Under 44.5", "SEA", "NE")
    assert _grade(over, "SEA", "NE", 27, 20) == "GANADA"
    assert _grade(under, "SEA", "NE", 20, 17) == "GANADA"


def test_integer_total_push():
    over = _parse_pick("Over 44", "SEA", "NE")
    assert _grade(over, "SEA", "NE", 24, 20) == "PUSH"
