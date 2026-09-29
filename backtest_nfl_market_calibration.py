"""Chronological, market-specific NFL probability recalibration research.

2022-23: generate expanding-window out-of-sample forecasts.
2023: fit candidate calibrators. 2024: choose by Brier, then log-loss.
2025: refit selected calibrator on 2022-24 OOS forecasts, evaluate ONCE.
2025 has already been examined in prior experiments: this is a diagnostic
confirmation, NOT a pristine untouched holdout.
No deployment, no bet execution, no claims of point-in-time odds.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine,feature_groups
from backtest_nfl_independent_market_odds import decimal,novig

EPS=1e-5
def clip(p):return np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def logit(p):
    p=clip(p)
    return np.log(p/(1-p))
def fit_calibrator(name,raw,y):
    raw=clip(raw);y=np.asarray(y,dtype=int)
    if name=="RAW":return lambda p:clip(p)
    if name=="SHRINK":
        base=float(y.mean())
        weights=np.linspace(0,1,41)
        w=min(weights,key=lambda a:brier_score_loss(y,clip(a*raw+(1-a)*base)))
        return lambda p:clip(w*clip(p)+(1-w)*base)
    if name=="PLATT":
        lr=LogisticRegression(C=1.0,max_iter=1000)
        lr.fit(logit(raw).reshape(-1,1),y)
        return lambda p:clip(lr.predict_proba(logit(p).reshape(-1,1))[:,1])
    raise ValueError(name)

def fit_residual_model(residuals):
    arr=np.asarray(residuals,dtype=float)
    arr=arr[np.isfinite(arr)]
    if len(arr)<100:raise ValueError("Insufficient prior OOS residuals")
    return arr
def residual_prob(pred,line,residuals,temperature=1.0):
    arr=np.asarray(residuals)
    # Scale historical residual distribution; no target-season outcome involved.
    return float(np.mean(float(pred)+temperature*arr>float(line)))

def oof(g,market,cols,years=(2022,2023,2024)):
    result={}
    for year in years:
        train=g[(g.season>=2021)&(g.season<year)]
        val=g[g.season==year].reset_index(drop=True)
        if len(train)<100 or len(val)<100:continue
        model=IndependentMarketEngine(market,cols).fit(train)
        pred=model.predict(val)
        result[year]=(val,pred,model.target(val).to_numpy())
    return result

def odds_offers(r,market,pred,residuals=None,temp=1):
    if market=="ML":
        ph=float(pred)
        return [("HOME",ph,r.home_moneyline,r.away_moneyline,int(r.margen_local>0),False),
                ("AWAY",1-ph,r.away_moneyline,r.home_moneyline,int(r.margen_local<0),False)]
    if market=="SPREAD":
        line=pd.to_numeric(r.spread_line,errors="coerce")
        if pd.isna(line):return []
        ph=residual_prob(pred,line,residuals,temp)
        pa=float(np.mean(float(pred)+temp*residuals<float(line)))
        push=abs(float(r.margen_local)-line)<1e-9
        return [("HOME",ph,r.home_spread_odds,r.away_spread_odds,int(r.margen_local>line),push),
                ("AWAY",pa,r.away_spread_odds,r.home_spread_odds,int(r.margen_local<line),push)]
    line=pd.to_numeric(r.total_line,errors="coerce")
    if pd.isna(line):return []
    po=residual_prob(pred,line,residuals,temp)
    pu=float(np.mean(float(pred)+temp*residuals<float(line)))
    push=abs(float(r.puntos_totales)-line)<1e-9
    return [("OVER",po,r.over_odds,r.under_odds,int(r.puntos_totales>line),push),
            ("UNDER",pu,r.under_odds,r.over_odds,int(r.puntos_totales<line),push)]

def market_eval(frame,pred,market,residuals=None,temp=1):
    rows=[]
    for i,r in frame.reset_index(drop=True).iterrows():
        for side,p,odd,other,win,push in odds_offers(r,market,pred[i],residuals,temp):
            d=decimal(odd);m=novig(odd,other)
            if d is None or m is None or push:continue
            rows.append({"game_id":r.game_id,"side":side,"prob":float(p),"win":int(win),
                         "market_prob":float(m),"odds":float(odd),
                         "edge_pp":100*(p-m),"ev_pct":100*(p*d-1),
                         "return":d-1 if win else -1.0})
    return pd.DataFrame(rows)

def metrics(df,market):
    if df.empty:return {"n_sides":0}
    # Two opposite sides from a single game are NOT independent observations.
    out={"n_sides":len(df),"n_games":int(df.game_id.nunique()),
         "brier":round(float(brier_score_loss(df.win,clip(df.prob))),5),
         "market_brier":round(float(brier_score_loss(df.win,clip(df.market_prob))),5),
         "logloss":round(float(log_loss(df.win,clip(df.prob))),5),
         "market_logloss":round(float(log_loss(df.win,clip(df.market_prob))),5),
         "mean_abs_market_gap_pp":round(float(np.mean(abs(df.prob-df.market_prob))*100),3)}
    sel=df[(df.prob>=.58)&(df.edge_pp>=3)&(df.ev_pct>=3)]
    if market=="ML":sel=sel[sel.odds<0]
    out["screen_n"]=len(sel)
    out["screen_roi"]=round(float(sel["return"].mean()),5) if len(sel) else None
    out["screen_pred"]=round(float(sel.prob.mean()),4) if len(sel) else None
    out["screen_actual"]=round(float(sel.win.mean()),4) if len(sel) else None
    return out

def main():
    root=Path("data")
    g=pd.read_csv(root/"historico_nfl_games.csv")
    p=pd.read_csv(root/"historico_nfl_pbp_team_game.csv")
    q=pd.read_csv(root/"historico_nfl_qbs.csv")
    g=g[pd.to_numeric(g.season,errors="coerce")<=2025]
    p=p[pd.to_numeric(p.season,errors="coerce")<=2025]
    q=q[pd.to_numeric(q.season,errors="coerce")<=2025]
    f=build_expanded_pregame(g,p,q)
    cols=["game_id","home_moneyline","away_moneyline","spread_line",
          "home_spread_odds","away_spread_odds","total_line","over_odds","under_odds"]
    f=f.merge(g[cols].drop_duplicates("game_id"),on="game_id",how="left",validate="one_to_one")
    groups=feature_groups(f)
    rows=[]
    for market,variant in (("ML","FULL"),("SPREAD","BASELINE"),("TOTAL","BASELINE")):
        features=groups[market][variant]
        history=oof(f,market,features)
        if not all(y in history for y in (2022,2023,2024)):
            raise SystemExit("Need 2022, 2023, 2024 OOS folds")
        val,pred24,y24=history[2024]
        if market=="ML":
            _,pred23,y23=history[2023]
            scores={}
            for name in ("RAW","SHRINK","PLATT"):
                calibrator=fit_calibrator(name,pred23,y23)
                calibrated=calibrator(pred24)
                scores[name]={"brier":float(brier_score_loss(y24,calibrated)),
                              "logloss":float(log_loss(y24,calibrated))}
            chosen=min(scores,key=lambda k:(scores[k]["brier"],scores[k]["logloss"]))
            fitpred=np.concatenate([history[y][1] for y in (2022,2023,2024)])
            fity=np.concatenate([history[y][2] for y in (2022,2023,2024)])
            calibrator=fit_calibrator(chosen,fitpred,fity)
            print("SELECTION",json.dumps({"market":market,"variant":variant,"chosen":chosen,
                   "validation_2024":scores}),flush=True)
        else:
            r22=fit_residual_model(history[2022][2]-history[2022][1])
            r23=fit_residual_model(history[2023][2]-history[2023][1])
            scores={}
            # Calibrate 2024 using only residuals from earlier years.
            for temp in (1.0,1.15,1.3,1.5,1.75,2.0):
                v=market_eval(val,pred24,market,r23,temp)
                scores[str(temp)]=metrics(v,market)
            chosen=min(scores,key=lambda k:(scores[k]["brier"],scores[k]["logloss"]))
            # Refit empirical residual distribution from strictly earlier OOS years.
            residuals=fit_residual_model(np.concatenate([history[y][2]-history[y][1]
                                                          for y in (2022,2023,2024)]))
            print("SELECTION",json.dumps({"market":market,"variant":variant,
                  "chosen_temperature":float(chosen),"validation_2024":scores,
                  "oos_residuals":len(residuals)}),flush=True)
        train=f[(f.season>=2021)&(f.season<2025)]
        test=f[f.season==2025].reset_index(drop=True)
        model=IndependentMarketEngine(market,features).fit(train)
        raw=model.predict(test)
        if market=="ML":
            for name,pred in (("RAW",raw),("CALIBRATED_"+chosen,calibrator(raw))):
                data=market_eval(test,pred,market)
                report={"market":market,"variant":variant,"method":name,**metrics(data,market)}
                rows.extend([{"market":market,"variant":variant,"method":name,**r}
                             for r in data.to_dict("records")])
                print("TEST_2025",json.dumps(report),flush=True)
        else:
            for name,temp in (("RAW",1.0),("CALIBRATED",float(chosen))):
                data=market_eval(test,raw,market,residuals,temp)
                report={"market":market,"variant":variant,"method":name,
                        "residual_temperature":temp,**metrics(data,market)}
                rows.extend([{"market":market,"variant":variant,"method":name,**r}
                             for r in data.to_dict("records")])
                print("TEST_2025",json.dumps(report),flush=True)
    pd.DataFrame(rows).to_csv("nfl_market_calibration_2025_diagnostic.csv",index=False)
    print("LIMITATION 2025 previously inspected; no pristine holdout. Odds and weather snapshot timestamps unverified.",flush=True)
if __name__=="__main__":main()
