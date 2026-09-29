"""Audit independent NFL market probability calibration and historical odds provenance.
No deployment, picks, staking, or modification of existing backtests.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine,feature_groups
from backtest_nfl_independent_market_odds import calibration,decimal,novig

def calibration_bins(frame):
    out=[]
    for lo,hi in ((0,.45),(.45,.50),(.50,.55),(.55,.60),(.60,.65),(.65,.70),(.70,.80),(.80,1.001)):
        s=frame[(frame.prob>=lo)&(frame.prob<hi)]
        if len(s):
            out.append({"bin":f"{lo:.2f}-{hi:.2f}","n":len(s),"mean_p":round(float(s.prob.mean()),4),
                        "actual":round(float(s.win.mean()),4),"gap_pp":round(float(100*(s.prob.mean()-s.win.mean())),2)})
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
    plans=feature_groups(f)
    train=f[(f.season>=2021)&(f.season<2025)]
    hold=f[f.season==2025].reset_index(drop=True)
    all_rows=[]
    for market,names in {"ML":["BASELINE","FULL"],"SPREAD":["BASELINE","BASE+team_pbp","FULL"],
                         "TOTAL":["BASELINE","BASE+qb_efficiency","FULL"]}.items():
        for name in names:
            if name not in plans[market]:continue
            model=IndependentMarketEngine(market,plans[market][name]).fit(train)
            preds=model.predict(hold)
            cal=calibration(f,market,plans[market][name])
            residuals=cal.get("residuals")
            for i,r in hold.iterrows():
                pred=float(preds[i])
                if market=="ML":
                    home=cal["shrink"]*pred+(1-cal["shrink"])*cal["base"]
                    offers=[("HOME",home,r.home_moneyline,r.away_moneyline,int(r.margen_local>0),False),
                            ("AWAY",1-home,r.away_moneyline,r.home_moneyline,int(r.margen_local<0),False)]
                elif market=="SPREAD":
                    line=pd.to_numeric(r.spread_line,errors="coerce")
                    if pd.isna(line):continue
                    home=float(np.mean(pred+residuals>line));away=float(np.mean(pred+residuals<line))
                    push=abs(float(r.margen_local)-line)<1e-9
                    offers=[("HOME",home,r.home_spread_odds,r.away_spread_odds,int(r.margen_local>line),push),
                            ("AWAY",away,r.away_spread_odds,r.home_spread_odds,int(r.margen_local<line),push)]
                else:
                    line=pd.to_numeric(r.total_line,errors="coerce")
                    if pd.isna(line):continue
                    over=float(np.mean(pred+residuals>line));under=float(np.mean(pred+residuals<line))
                    push=abs(float(r.puntos_totales)-line)<1e-9
                    offers=[("OVER",over,r.over_odds,r.under_odds,int(r.puntos_totales>line),push),
                            ("UNDER",under,r.under_odds,r.over_odds,int(r.puntos_totales<line),push)]
                for side,prob,odd,other,win,push in offers:
                    d=decimal(odd);m=novig(odd,other)
                    if d is None or m is None or push or not np.isfinite(prob):continue
                    all_rows.append({"market":market,"variant":name,"game_id":r.game_id,
                       "side":side,"prob":float(prob),"market_prob":float(m),"odds":float(odd),
                       "win":win,"edge_pp":100*(prob-m),"ev_pct":100*(prob*d-1),
                       "return":d-1 if win else -1.0})
    out=pd.DataFrame(all_rows)
    out.to_csv("nfl_independent_probability_audit.csv",index=False)
    for (market,variant),part in out.groupby(["market","variant"]):
        # Both sides per game: useful for calibration, not independent observations.
        selected=part[(part.prob>=.55)&(part.edge_pp>=3)&(part.ev_pct>=3)].copy()
        # Production ML additionally requires negative American odds.
        production=selected[selected.odds<0] if market=="ML" else selected
        report={"market":market,"variant":variant,"n_sides":len(part),
           "brier":round(float(brier_score_loss(part.win,part.prob)),4),
           "mean_pred":round(float(part.prob.mean()),4),
           "mean_actual":round(float(part.win.mean()),4),
           "mean_abs_market_gap_pp":round(float(np.mean(abs(part.prob-part.market_prob))*100),2),
           "selected_n":len(selected),
           "selected_roi":round(float(selected["return"].mean()),4) if len(selected) else None,
           "production_like_n":len(production),
           "production_like_roi":round(float(production["return"].mean()),4) if len(production) else None,
           "selected_pred":round(float(selected.prob.mean()),4) if len(selected) else None,
           "selected_actual":round(float(selected.win.mean()),4) if len(selected) else None,
           "bins":calibration_bins(part)}
        if market=="ML":
            report["log_loss"]=round(float(log_loss(part.win,np.clip(part.prob,1e-6,1-1e-6))),4)
        print("AUDIT",json.dumps(report),flush=True)
    # Data provenance: nflverse schedules do not expose historical pregame snapshots
    # merely by carrying home_moneyline/spread_line/weather columns.
    print("PROVENANCE",json.dumps({"odds_snapshot_verified":False,
          "weather_forecast_timestamp_verified":False,
          "qb_starter_confirmed_pregame":False,
          "note":"No bettable historical edge may be claimed without timestamped bookmaker and forecast records."}),flush=True)
    print("CAUTION: selection thresholds evaluated on 2025 are diagnostics, not optimized strategy.",flush=True)
if __name__=="__main__":main()
