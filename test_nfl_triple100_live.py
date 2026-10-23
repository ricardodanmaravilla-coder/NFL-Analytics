import pandas as pd
from modules import nfl_therundown_odds as odds
from modules import nfl_triple100_live as live

class Response:
    def __init__(self, status): self.status_code = status
    def json(self): return {"events":[]}

def test_three_keys_rotate_on_quota(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY","key-one")
    monkeypatch.setenv("THERUNDOWN_KEY_2","key-two")
    monkeypatch.setenv("THERUNDOWN_KEY_3","key-three")
    attempted=[]
    def fake_get(url,**kw):
        attempted.append(kw["headers"]["X-TheRundown-Key"])
        return Response(429 if len(attempted)<3 else 200)
    response, _, _ = odds._request_snapshot("2026-10-26",get_fn=fake_get)
    assert response.status_code==200
    assert attempted==["key-one","key-two","key-three"]

def test_three_keys_deduplicate(monkeypatch):
    monkeypatch.setenv("THERUNDOWN_KEY","same")
    monkeypatch.setenv("THERUNDOWN_KEY_2","same")
    monkeypatch.setenv("THERUNDOWN_KEY_3","third")
    assert odds._keys()==["same","third"]

def test_missing_quotes_no_fabricated_markets(monkeypatch):
    monkeypatch.setattr(live,"configured",lambda:True)
    monkeypatch.setattr(live,"get_markets",lambda h,a,d:None)
    r=live.scan_live(pd.DataFrame(),[{"home_team":"PHI","away_team":"DAL","gameday":"2026-10-26"}])
    assert r["checked"]==0 and not r["approved"] and len(r["unpriced_games"])==1

def test_triple100_with_complete_mocked_quotes(monkeypatch):
    df=pd.DataFrame([
        {"gameday":f"202{i}-10-01","home_team":"PHI","away_team":"DAL",
         "home_score":28,"away_score":14,"game_type":"REG"}
        for i in range(4)])
    monkeypatch.setattr(live,"configured",lambda:True)
    monkeypatch.setattr(live,"get_markets",lambda h,a,d:{
        "home_moneyline":-150,"away_moneyline":130,
        "home_spread":-6.5,"away_spread":6.5,
        "home_spread_odds":-110,"away_spread_odds":-110,
        "total_line":40.5,"over_odds":-110,"under_odds":-110,
        "source":"TheRundown","book":"Test Book",
        "spread_book":"Test Book","total_book":"Test Book"})
    r=live.scan_live(df,[{"home_team":"PHI","away_team":"DAL","gameday":"2026-10-26"}])
    assert r["checked"]==6
    assert len(r["approved"])==3
    assert {x["market"] for x in r["approved"]}=={"moneyline","spread","total_over"}
