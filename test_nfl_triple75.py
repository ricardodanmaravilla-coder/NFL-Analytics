import pandas as pd
from modules.nfl_triple100_fullgame import evaluate

def games(wins=3):
    rows=[]
    for i in range(4):
        win=i<wins
        rows.append({"gameday":f"202{i}-10-01", "home_team":"DET",
                     "away_team":"MIN", "home_score":28 if win else 10,
                     "away_score":14 if win else 17,"game_type":"REG"})
    return pd.DataFrame(rows)

def test_exact_75_accepted_all_three():
    r=evaluate(games(),team="DET",opponent="MIN",
               kickoff="2026-10-26",venue="home",market="moneyline")
    assert r["triple_75"] is True
    assert r["triple_80"] is False
    assert r["minimum_rate"] == 0.75
    assert all(x["games"] >= 4 for x in r["samples"].values())

def test_50_rejected():
    r=evaluate(games(2),team="DET",opponent="MIN",
               kickoff="2026-10-26",venue="home",market="moneyline")
    assert r["triple_75"] is False

def test_push_not_counted():
    df=games()
    df.loc[0,"home_score"]=14
    r=evaluate(df,team="DET",opponent="MIN",kickoff="2026-10-26",
               venue="home",market="moneyline")
    assert r["triple_75"] is False
