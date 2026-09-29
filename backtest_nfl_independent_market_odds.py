"""Odds-aware locked 2025 evaluation of three independent NFL research engines.
2023/24 expanding-year predictions calibrate ML and provide independent MC residuals.
2025 outcomes are never used in model selection or calibration.
This is a research backtest, NOT a production betting strategy.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine, feature_groups

SEED=1729
def decimal(american):
    try:
        x=float(american)
        if not np.isfinite(x) or x==0:return None
        return 1+x/100 if x>0 else 1+100/abs(x)
    except (ValueError,TypeError):return None

def novig(a,b):
    da,db=decimal(a),decimal(b)
    if da is None or db is None:return None
    return (1/da)/((1/da)+(1/db))

def calibration(g,market,cols):
    residuals=[]; probs=[]; labels=[]
    for year in (2023,2024):
        train=g[(g.season>=2021)&(g.season<year)]
        val=g[g.season==year]
        model=IndependentMarketEngine(market,cols).fit(train)
        pred=model.predict(val)
        if market=="ML":
            probs.extend(pred.tolist());labels.extend(model.target(val).tolist())
        else:
            residuals.extend((model.target(val).to_numpy()-pred).tolist())
    if market=="ML":
        # Conservative one-parameter correction: shrink to prior validation base rate.
        # 2025 never used in calibration.
        p=np.asarray(probs);y=np.asarray(labels)
        if len(p)<100:return {"shrink":1.0,"base":0.5}
        base=float(np.mean(y))
        from sklearn.metrics import log_loss
        options=np.linspace(0.0,1.0,21)
        best=min(options,key=lambda s:log_loss(y,np.clip(s*p+(1-s)*base,1e-5,1-1e-5)))
        return {"shrink":float(best),"base":base}
    arr=np.asarray(residuals,dtype=float)
    return {"residuals":arr[np.isfinite(arr)]}

def row_result(g,market,variant,side,prob,odd,other,win,push=False):
    d=decimal(odd);m=novig(odd,other)
    if d is None or m is None or not np.isfinite(prob):return None
    edge=100*(prob-m);ev=100*(prob*d-1)
    # Fixed predeclared screening for diagnostic comparison, not an optimized rule.
    if prob<0.55 or edge<0.03*100 or ev<3 or push:return None
    return {"game_id":g.game_id,"market":market,"variant":variant,"side":side,
            "probability":round(float(prob),5),"market_probability":round(float(m),5),
            "edge_pp":round(edge,3),"ev_pct":round(ev,3),"odds":odd,
            "win":int(win),"return":d-1 if win else -1.0}

def main():
    root=Path("data")
    g=pd.read_csv(root/"historico_nfl_games.csv")
    p=pd.read_csv(root/"historico_nfl_pbp_team_game.csv")
    q=pd.read_csv(root/"historico_nfl_qbs.csv")
    g=g[pd.to_numeric(g.season,errors="coerce")<=2025]
    p=p[pd.to_numeric(p.season,errors="coerce")<=2025]
    q=q[pd.to_numeric(q.season,errors="coerce")<=2025]
    f=build_expanded_pregame(g,p,q)
    odds_cols=["game_id","home_moneyline","away_moneyline","spread_line",
               "home_spread_odds","away_spread_odds","total_line","over_odds","under_odds"]
    if any(c not in g.columns for c in odds_cols):raise SystemExit("Missing odds columns")
    # Historical game_id unique is essential for leakage-safe odds merge.
    odds=g[odds_cols].drop_duplicates("game_id")
    f=f.merge(odds,on="game_id",how="left",validate="one_to_one")
    train=f[(f.season>=2021)&(f.season<2025)]
    hold=f[f.season==2025].reset_index(drop=True)
    groups=feature_groups(f)
    rows=[]
    variants={
        "ML":("BASELINE","FULL"),
        "SPREAD":("BASELINE","BASE+team_pbp","FULL"),
        "TOTAL":("BASELINE","BASE+qb_efficiency","FULL"),
    }
    for market,names in variants.items():
        for name in names:
            if name not in groups[market]:continue
            cols=groups[market][name]
            cal=calibration(f,market,cols)
            model=IndependentMarketEngine(market,cols).fit(train)
            predictions=model.predict(hold)
            if market=="ML":
                probabilities=cal["shrink"]*predictions+(1-cal["shrink"])*cal["base"]
                print("CALIBRATION",json.dumps({"market":market,"variant":name,**cal}),flush=True)
            else:
                residuals=cal["residuals"]
                if len(residuals)<100:raise SystemExit("Insufficient independent OOS residuals")
                print("CALIBRATION",json.dumps({"market":market,"variant":name,
                      "oos_residuals":len(residuals),"residual_sd":float(np.std(residuals))}),flush=True)
            for i,r in hold.iterrows():
                pred=float(predictions[i])
                if market=="ML":
                    ph=float(probabilities[i]);pa=1-ph
                    offers=[("HOME",ph,r.home_moneyline,r.away_moneyline,r.margen_local>0),
                            ("AWAY",pa,r.away_moneyline,r.home_moneyline,r.margen_local<0)]
                elif market=="SPREAD":
                    line=pd.to_numeric(r.spread_line,errors="coerce")
                    if pd.isna(line):continue
                    # nflverse spread_line: positive means home favorite;
                    # home spread is -spread_line.
                    threshold=float(line)
                    ph=float(np.mean(pred+residuals>threshold))
                    pa=float(np.mean(pred+residuals<threshold))
                    push=abs(float(r.margen_local)-threshold)<1e-9
                    offers=[("HOME",ph,r.home_spread_odds,r.away_spread_odds,
                             r.margen_local>threshold),
                            ("AWAY",pa,r.away_spread_odds,r.home_spread_odds,
                             r.margen_local<threshold)]
                else:
                    line=pd.to_numeric(r.total_line,errors="coerce")
                    if pd.isna(line):continue
                    po=float(np.mean(pred+residuals>line))
                    pu=float(np.mean(pred+residuals<line))
                    push=abs(float(r.puntos_totales)-float(line))<1e-9
                    offers=[("OVER",po,r.over_odds,r.under_odds,r.puntos_totales>line),
                            ("UNDER",pu,r.under_odds,r.over_odds,r.puntos_totales<line)]
                for side,prob,odd,other,win in offers:
                    item=row_result(r,market,name,side,prob,odd,other,win,
                                    push if market!="ML" else False)
                    if item is not None:rows.append(item)
            subset=[x for x in rows if x["market"]==market and x["variant"]==name]
            report={"market":market,"variant":name,"n":len(subset)}
            if subset:
                report.update({"winrate":float(np.mean([x["win"] for x in subset])),
                               "roi":float(np.mean([x["return"] for x in subset])),
                               "mean_edge_pp":float(np.mean([x["edge_pp"] for x in subset]))})
            print("MARKET_REPORT",json.dumps(report),flush=True)
    pd.DataFrame(rows).to_csv("nfl_independent_market_odds_results.csv",index=False)
    print("WARNING: historical closing odds/weather may not equal timestamped pregame availability.",flush=True)
if __name__=="__main__":main()
