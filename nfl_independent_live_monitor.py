"""Live PAPER-ONLY recommendations from three genuinely independent NFL engines.
ML FULL RF classifier, SPREAD BASE RF regressor, TOTAL BASE RF regressor.
No production API scan, no wagers, no Sheets writes, no artificial prices.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import nfl_data_py as nfl
import requests
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine,feature_groups
from nfl_shadow_monitor import ROOT,BASE,save_jsonl,read,utc,grade

VARIANTS={"ML":"FULL","SPREAD":"BASELINE","TOTAL":"BASELINE"}
TEMPERATURE={"SPREAD":2.0,"TOTAL":2.0}
def odds_decimal(x):
    if x is None:return None
    x=float(x)
    return 1+x/100 if x>=100 else (1+100/abs(x) if x<=-100 else None)
def paper_kelly(p, odd, bankroll=5000.0):
    """Hypothetical staking only: quarter-Kelly, clamped to 3%-10% for recommended picks."""
    d=odds_decimal(odd)
    if d is None:return {"kelly_full_pct":0.0,"kelly_quarter_pct":0.0,"kelly_assigned_pct":0.0,"paper_stake":0.0}
    b=d-1.0
    full=max(0.0,(b*p-(1-p))/b)
    quarter=full/4.0
    assigned=min(0.10,max(0.03,quarter))
    return {"kelly_full_pct":round(full*100,2),"kelly_quarter_pct":round(quarter*100,2),
            "kelly_assigned_pct":round(assigned*100,2),"paper_stake":round(bankroll*assigned,2)}

def offer(side,p,odd,other,line=None):
    d=odds_decimal(odd);e=odds_decimal(other)
    if d is None or e is None:return None
    market=(1/d)/(1/d+1/e)
    return {"side":side,"probability":round(100*p,2),"market_probability":round(100*market,2),
            "odds":int(odd),"line":line,"edge_pp":round(100*(p-market),2),
            "ev_pct":round(100*(p*d-1),2)}
def history_and_future(season,week):
    g=pd.read_csv("data/historico_nfl_games.csv")
    p=pd.read_csv("data/historico_nfl_pbp_team_game.csv")
    q=pd.read_csv("data/historico_nfl_qbs.csv")
    for df in (g,p,q):
        df["season"]=pd.to_numeric(df.season,errors="coerce")
        df["week"]=pd.to_numeric(df.week,errors="coerce")
    g=g[(g.season<season)|((g.season==season)&(g.week<week))]
    p=p[(p.season<season)|((p.season==season)&(p.week<week))]
    q=q[(q.season<season)|((q.season==season)&(q.week<week))]
    # Only completed historical games enter training.
    g=g.dropna(subset=["home_score","away_score"])
    sched=nfl.import_schedules([season])
    upcoming=sched[(sched.week==week)&(sched.home_score.isna())].copy()
    if "game_type" in upcoming:upcoming=upcoming[upcoming.game_type.isin(["REG","WC","DIV","CON","SB"])]
    if upcoming.empty:return None,None,None
    # Incorporate verified final scores from the CURRENT season before target week. 
    # Archived CSVs may stop at the prior season; without this, rolling team 
    # form remains stale for every 2026 game. PBP/QB remain explicitly stale 
    # until separate current-season feeds are available.
    completed=sched[(sched.week<week)&sched.home_score.notna()&sched.away_score.notna()].copy()
    if not completed.empty:
        g=pd.concat([g,completed],ignore_index=True,sort=False)
        g=g.drop_duplicates("game_id",keep="last") 
    # Synthetic zero targets exist ONLY for feature generation and are never
    # passed to model.fit or calibration. All earlier-week history is real.
    future=upcoming.copy()
    future["home_score"]=0.0;future["away_score"]=0.0
    combined=pd.concat([g,future],ignore_index=True,sort=False)
    f=build_expanded_pregame(combined,p,q)
    training=f[(f.season<season)|((f.season==season)&(f.week<week))].copy()
    test=f[(f.season==season)&(f.week==week)].copy()
    if training.empty or test.empty:raise RuntimeError("Insufficient historical/pregame features")
    return training,test,upcoming
def residuals_for(market,features,training):
    # Historical expanding OOS folds. Current season outcomes are never used
    # to calibrate the same season; conservative frozen 2022-24 residual pool.
    errors=[]
    for y in (2022,2023,2024):
        tr=training[(training.season>=2021)&(training.season<y)]
        te=training[training.season==y]
        if len(tr)<100 or len(te)<100:continue
        model=IndependentMarketEngine(market,features).fit(tr)
        errors.extend((model.target(te).to_numpy()-model.predict(te)).tolist())
    arr=np.asarray(errors,dtype=float)
    arr=arr[np.isfinite(arr)]
    if len(arr)<100:raise RuntimeError(f"{market}: insufficient prior OOS residuals")
    return arr
def independent_scan(season,week):
    training,test,upcoming=history_and_future(season,week)
    if test is None:return {"season":season,"week":week,"recommendations":[],"diagnostics":["NO_FUTURE_GAMES"]}
    r=requests.get(f"{BASE}/api/slate/{season}/{week}",timeout=120)
    r.raise_for_status()
    slate=r.json()
    if slate.get("odds_provider")!="TheRundown":raise RuntimeError("Real TheRundown odds required")
    quotes={g["game"]:g for g in slate.get("games",[])}
    plans=feature_groups(training)
    predictions={}
    for market,variant in VARIANTS.items():
        cols=plans[market][variant]
        engine=IndependentMarketEngine(market,cols)
        engine.fit(training[training.season>=2021])
        preds=engine.predict(test)
        errors=residuals_for(market,cols,training) if market!="ML" else None
        predictions[market]=(preds,errors,len(cols))
    stamp=utc();out=[];diagnostics=[]
    for i,g in test.reset_index(drop=True).iterrows():
        game=f"{g.away_team} @ {g.home_team}"
        quote=quotes.get(game)
        if not quote:
            diagnostics.append({"game":game,"status":"NO_REAL_ODDS"});continue
        available=0
        for market in VARIANTS:
            preds,errors,nfeatures=predictions[market]
            pred=float(preds[i]);offers=[]
            if market=="ML":
                ph=float(np.clip(pred,0,1))
                if quote.get("home_moneyline") is not None and quote.get("away_moneyline") is not None:
                    offers=[offer(g.home_team,ph,quote["home_moneyline"],quote["away_moneyline"]),
                            offer(g.away_team,1-ph,quote["away_moneyline"],quote["home_moneyline"])]
            elif market=="SPREAD":
                hline=quote.get("home_spread");aline=quote.get("away_spread")
                ho=quote.get("home_spread_odds");ao=quote.get("away_spread_odds")
                if None not in (hline,aline,ho,ao):
                    # Home cover: home margin + home handicap > 0.
                    draws=pred+TEMPERATURE[market]*errors
                    ph=float(np.mean(draws>-float(hline)))
                    pa=float(np.mean(draws<-float(hline)))
                    offers=[offer(g.home_team,ph,ho,ao,hline),offer(g.away_team,pa,ao,ho,aline)]
            else:
                line=quote.get("total_line");oo=quote.get("over_odds");uo=quote.get("under_odds")
                if None not in (line,oo,uo):
                    draws=pred+TEMPERATURE[market]*errors
                    po=float(np.mean(draws>float(line)));pu=float(np.mean(draws<float(line)))
                    offers=[offer("Over",po,oo,uo,line),offer("Under",pu,uo,oo,line)]
            for x in offers:
                if x is None:continue
                available+=1
                x.update({"game":game,"game_id":g.game_id,"season":season,"week":week,
                          "market":market,"model":f"independent_{market.lower()}_{VARIANTS[market].lower()}_v1",
                          "feature_count":nfeatures,"prediction":round(pred,3),
                          "calibration":"RAW" if market=="ML" else "OOS_RESIDUAL_X2",
                          "book":quote.get({"ML":"book","SPREAD":"spread_book","TOTAL":"total_book"}[market]),
                          "odds_source":quote.get("odds_source"),"odds_fetched_at":quote.get("odds_fetched_at"),
                          "captured_at":stamp,"mode":"PAPER_ONLY","stake":0})
                # All model recommendations above threshold are logged, not only
                # apparent positive ROI; every quoted side is saved as a snapshot.
                x["recommended"]=x["probability"]>=58 and x["edge_pp"]>=3 and x["ev_pct"]>=3
                if x["recommended"]:
                    x.update(paper_kelly(x["probability"]/100.0,x["odds"]))
                else:
                    x.update({"kelly_full_pct":0.0,"kelly_quarter_pct":0.0,"kelly_assigned_pct":0.0,"paper_stake":0.0})
                x["pick"]=(f"{x['side']} ML" if market=="ML" else
                           f"{x['side']} {float(x['line']):+g}" if market=="SPREAD" else
                           f"{x['side']} {float(x['line']):g}")
                out.append(x)
        diagnostics.append({"game":game,"status":"QUOTED" if available else "NO_COMPLETE_MARKET",
                            "quoted_sides":available})
    return {"season":season,"week":week,"captured_at":stamp,
            "provenance":{"current_season_final_scores_included":True,
             "current_season_qb_pbp_verified":False,"pregame_weather_verified":False,
             "odds_source":"TheRundown","money_staked":0},
            "recommendations":out,"diagnostics":diagnostics}
def persist(result):
    season,week=result["season"],result["week"]
    path=ROOT/"independent_recommendations.jsonl"
    prior={r["recommendation_id"] for r in read(path)}
    new=[]
    for r in result["recommendations"]:
        if not r["recommended"]:continue
        # Freeze first observed quote/line per model, game and side.
        rid="|".join(map(str,(season,week,r["model"],r["game"],r["side"])))
        if rid in prior:continue
        prior.add(rid);new.append(dict(r,recommendation_id=rid,status="PENDING"))
    save_jsonl(ROOT/"independent_snapshots.jsonl",[result])
    save_jsonl(path,new)
    return {"new":len(new),"total":len(prior),"quoted_sides":len(result["recommendations"]),
            "diagnostics":result["diagnostics"]}
def settle():
    pending=[r for r in read(ROOT/"independent_recommendations.jsonl")
             if r["recommendation_id"] not in {x["recommendation_id"] for x in read(ROOT/"independent_results.jsonl")}]
    if not pending:return 0
    seasons=sorted({r["season"] for r in pending})
    games=pd.concat([nfl.import_schedules([s]) for s in seasons],ignore_index=True)
    results={}
    for _,g in games.iterrows():
        if pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")) and pd.notna(g.get("result")):
            results[(int(g.season),int(g.week),f"{g.away_team} @ {g.home_team}")]=(float(g.home_score),float(g.away_score))
    done=[]
    for r in pending:
        score=results.get((r["season"],r["week"],r["game"]))
        if score is None:continue
        result=grade(r,*score);d=odds_decimal(r["odds"])
        done.append({"recommendation_id":r["recommendation_id"],"settled_at":utc(),
                     "home_score":score[0],"away_score":score[1],"result":result,
                     "paper_return_per_unit":0 if result=="PUSH" else round(d-1 if result=="WIN" else -1,6)})
    save_jsonl(ROOT/"independent_results.jsonl",done)
    return len(done)
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--season",type=int);ap.add_argument("--week",type=int)
    a=ap.parse_args()
    n=settle()
    if a.season and a.week:pair=(a.season,a.week)
    else:
        from nfl_shadow_monitor import current_week,schedule
        pair=current_week(schedule())
    if pair:
        report=persist(independent_scan(*pair))
        print(json.dumps({"settled":n,**report},default=str))
    else:print(json.dumps({"settled":n,"status":"NO_UPCOMING_WEEK"}))
