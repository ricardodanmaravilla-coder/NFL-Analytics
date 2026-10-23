import pandas as pd
import pytest
from modules.nfl_triple100_fullgame import evaluate

def sample():
    rows=[]
    # Four A-B games, A always wins by at least 7, all A home;
    # these games intentionally overlap across all three trend groups.
    for i in range(4):
        rows.append(dict(gameday=f"202{i}-10-01", home_team="AAA",
                         away_team="BBB", home_score=28, away_score=14,
                         game_type="REG"))
    return pd.DataFrame(rows)

def test_perfect_overlap_allowed():
    r=evaluate(sample(),team="AAA",opponent="BBB",kickoff="2026-10-26",
               venue="home",market="spread",line=-6.5)
    assert r["triple_100"]
    assert [r["samples"][k]["games"] for k in ("general","h2h","venue")] == [4,4,4]

def test_all_markets():
    for market,line in (("moneyline",None),("spread",-6.5),("total_over",40.5),("total_under",50.5)):
        assert evaluate(sample(),team="AAA",opponent="BBB",kickoff="2026-10-26",
                        venue="home",market=market,line=line)["triple_100"]

def test_minimum_strict_all_three():
    df=sample().iloc[:3]
    r=evaluate(df,team="AAA",opponent="BBB",kickoff="2026-10-26",
               venue="home",market="moneyline")
    assert not r["triple_100"]

def test_push_is_not_win():
    r=evaluate(sample(),team="AAA",opponent="BBB",kickoff="2026-10-26",
               venue="home",market="spread",line=-14)
    assert not r["triple_100"]
    assert r["samples"]["general"]["pushes"]==4

def test_no_future_leak():
    r=evaluate(sample(),team="AAA",opponent="BBB",kickoff="2022-01-01",
               venue="home",market="moneyline")
    assert not r["triple_100"]

def test_no_cross_team_venue():
    r=evaluate(sample(),team="AAA",opponent="BBB",kickoff="2026-10-26",
               venue="away",market="moneyline")
    assert not r["triple_100"]

def test_reject_relaxed_minimum():
    with pytest.raises(ValueError):
        evaluate(sample(),team="AAA",opponent="BBB",kickoff="2026-10-26",
                 venue="home",market="moneyline",min_games=3)

def test_eighty_percent_accepts_four_of_five():
    df=sample()
    df=pd.concat([df,pd.DataFrame([dict(gameday="2025-10-01",home_team="AAA",away_team="BBB",home_score=10,away_score=17,game_type="REG")])],ignore_index=True)
    r=evaluate(df,team="AAA",opponent="BBB",kickoff="2026-10-26",venue="home",market="moneyline")
    assert r["triple_80"] is True
    assert r["triple_100"] is False
    assert all(v["wins"] == 4 and v["games"] == 5 for v in r["samples"].values())

def test_seventy_five_percent_rejected():
    df=sample().copy()
    df.loc[0,"home_score"]=10
    r=evaluate(df,team="AAA",opponent="BBB",kickoff="2026-10-26",venue="home",market="moneyline")
    assert r["triple_80"] is False
