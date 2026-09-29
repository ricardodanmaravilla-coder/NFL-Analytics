"""Append-only NFL shadow monitor. No betting, no Google Sheets writes.
Fetches the read-only Cloud Run shadow endpoint; commits immutable snapshots
and separate result records. Refuses to treat absent real odds as picks.
"""
import argparse,json,os,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl
import requests

ROOT=Path("data/nfl_shadow")
BASE=os.getenv("NFL_SHADOW_API","https://nfl-analytics-664101772310.northamerica-south1.run.app").rstrip("/")
def utc():return datetime.now(timezone.utc).isoformat()
def save_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return 0
    with path.open("a",encoding="utf8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)+"\n")
    return len(rows)
def read(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
def slate(season,week):
    url=f"{BASE}/api/shadow/{season}/{week}"
    r=requests.get(url,timeout=150)
    r.raise_for_status()
    data=r.json()
    if data.get("shadow") is not True:raise RuntimeError("Refusing non-shadow endpoint: would risk side effects")
    if not data.get("sheet_sync",{}).get("ok"):raise RuntimeError("Shadow API not healthy")
    return data
def schedule():
    year=datetime.now(timezone.utc).year
    games=nfl.import_schedules([year])
    games=games[games.game_type.isin(["REG","WC","DIV","CON","SB"])]
    games["gameday"]=pd.to_datetime(games.gameday,errors="coerce",utc=True)
    return games
def current_week(games):
    now=pd.Timestamp.now(tz="UTC")
    upcoming=games[(games.gameday>=now-pd.Timedelta(days=1))&
                   (games.gameday<=now+pd.Timedelta(days=9))&
                   (games.home_score.isna())]
    if upcoming.empty:return None
    # Choose nearest scheduled future week, not a week with already final games.
    g=upcoming.sort_values("gameday").iloc[0]
    return int(g.season),int(g.week)
def monitor(season=None,week=None):
    games=schedule()
    pair=(int(season),int(week)) if season and week else current_week(games)
    if pair is None:
        print("NO_UPCOMING_WEEK");return
    season,week=pair
    data=slate(season,week)
    stamp=utc()
    raw=ROOT/"snapshots.jsonl"
    records=ROOT/"recommendations.jsonl"
    existing={x["recommendation_id"] for x in read(records)}
    new=[]
    for p in data.get("bets",[])+data.get("leans",[]):
        if p.get("odds_source")!="TheRundown" or not p.get("odds_fetched_at"):
            continue
        market=p.get("market")
        if market not in ("ML","SPREAD","TOTAL"):continue
        # First observable recommendation for each game/market/side/line is frozen.
        rid=f"{season}|{week}|{p.get('game')}|{market}|{p.get('pick')}"
        if rid in existing:continue
        existing.add(rid)
        new.append({"recommendation_id":rid,"captured_at":stamp,"season":season,"week":week,
                    "model":"production_shadow_v1","game":p.get("game"),
                    "market":market,"pick":p.get("pick"),"line":p.get("line"),
                    "probability":p.get("probability"),"mc_probability":p.get("mc_probability"),
                    "odds":p.get("odds"),"edge_pp":p.get("edge"),"ev_pct":p.get("ev"),
                    "book":p.get("book"),"odds_source":p.get("odds_source"),
                    "odds_fetched_at":p.get("odds_fetched_at"),"status":"PENDING",
                    "stake":0,"mode":"PAPER_ONLY"})
    # Full scans are retained, including NO_ODDS diagnostics, for missing-data audits.
    save_jsonl(raw,[{"captured_at":stamp,"season":season,"week":week,
                     "recommendations":data.get("bets",[])+data.get("leans",[]),
                     "diagnostics":data.get("diagnostics",[])}])
    save_jsonl(records,new)
    print(json.dumps({"season":season,"week":week,"new":len(new),
                      "total":len(existing),"diagnostics":len(data.get("diagnostics",[]))}))
def grade(p,hs,aws):
    pick=str(p["pick"]);home=p["game"].split(" @ ")[-1];away=p["game"].split(" @ ")[0]
    if p["market"]=="ML":
        if hs==aws:return "PUSH"
        return "WIN" if pick.split(" ML")[0]==(home if hs>aws else away) else "LOSS"
    if p["market"]=="SPREAD":
        team=pick.split(" ")[0];line=float(p["line"])
        margin=(hs-aws) if team==home else (aws-hs)
        diff=margin+line
    else:
        line=float(p["line"]);total=hs+aws
        diff=total-line if pick.startswith("Over") else line-total
    return "PUSH" if abs(diff)<1e-9 else ("WIN" if diff>0 else "LOSS")
def settle():
    records=read(ROOT/"recommendations.jsonl")
    done={x["recommendation_id"] for x in read(ROOT/"results.jsonl")}
    pending=[x for x in records if x["recommendation_id"] not in done]
    if not pending:print("NO_PENDING");return
    seasons=sorted({int(x["season"]) for x in pending})
    schedules=pd.concat([nfl.import_schedules([s]) for s in seasons],ignore_index=True)
    by_game={}
    for _,g in schedules.iterrows():
        if pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")):
            key=(int(g.season),int(g.week),str(g.away_team)+" @ "+str(g.home_team))
            by_game[key]=(float(g.home_score),float(g.away_score))
    out=[]
    for p in pending:
        scores=by_game.get((p["season"],p["week"],p["game"]))
        if not scores:continue
        hs,aws=scores
        result=grade(p,hs,aws)
        odd=float(p["odds"]);d=1+odd/100 if odd>0 else 1+100/abs(odd)
        profit=0 if result=="PUSH" else (d-1 if result=="WIN" else -1)
        out.append({"recommendation_id":p["recommendation_id"],"settled_at":utc(),
                    "home_score":hs,"away_score":aws,"result":result,
                    "paper_return_per_unit":round(profit,6)})
    save_jsonl(ROOT/"results.jsonl",out)
    print(json.dumps({"settled":len(out),"pending":len(pending)-len(out)}))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--season",type=int);parser.add_argument("--week",type=int)
    args=parser.parse_args()
    # Settlement always runs even when upcoming games are not available.
    settle()
    monitor(args.season,args.week)
