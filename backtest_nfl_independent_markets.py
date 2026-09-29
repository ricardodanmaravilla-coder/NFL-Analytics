"""2023-24 chronological market-specific feature selection; 2025 locked test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,mean_absolute_error
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine,feature_groups,select_market_features

def main():
    root=Path("data")
    g=pd.read_csv(root/"historico_nfl_games.csv")
    p=pd.read_csv(root/"historico_nfl_pbp_team_game.csv")
    q=pd.read_csv(root/"historico_nfl_qbs.csv")
    g=g[pd.to_numeric(g.season,errors="coerce")<=2025]
    p=p[pd.to_numeric(p.season,errors="coerce")<=2025]
    q=q[pd.to_numeric(q.season,errors="coerce")<=2025]
    f=build_expanded_pregame(g,p,q)
    plans=feature_groups(f);out=[]
    for market,candidates in plans.items():
        # All selection finishes before 2025 outcomes are accessed.
        selection={}
        for year in (2023,2024):
            train=f[(f.season>=2021)&(f.season<year)]
            val=f[f.season==year]
            _,scores=select_market_features(train,val,market,candidates)
            selection[year]=scores
        base=np.mean([selection[y]["BASELINE"] for y in (2023,2024)])
        threshold=0.005 if market=="ML" else 0.10
        eligible=[name for name in candidates if name!="BASELINE" and
                  all(selection[y]["BASELINE"]-selection[y][name]>=threshold for y in (2023,2024))]
        chosen=min(eligible,key=lambda n:(np.mean([selection[y][n] for y in (2023,2024)]),
                                         len(candidates[n]))) if eligible else "BASELINE"
        print("SELECTION",json.dumps({"market":market,"chosen":chosen,"baseline_score":base,
             "candidate_scores":selection}),flush=True)
        train=f[(f.season>=2021)&(f.season<2025)]
        holdout=f[f.season==2025]
        for name in sorted(set(("BASELINE",chosen,"FULL"))):
            engine=IndependentMarketEngine(market,candidates[name]).fit(train)
            pred=engine.predict(holdout)
            y=engine.target(holdout).to_numpy()
            result={"market":market,"variant":name,"year":2025,"n":len(y),
                    "features":len(engine.usable)}
            if market=="ML":
                result.update({"log_loss":float(log_loss(y,np.clip(pred,1e-6,1-1e-6))),
                    "brier":float(np.mean((pred-y)**2)),
                    "accuracy":float(np.mean((pred>=0.5)==y))})
            else:
                result.update({"mae":float(mean_absolute_error(y,pred))})
                if market=="SPREAD":result["winner_accuracy"]=float(np.mean((pred>0)==(y>0)))
            out.append(result);print("HOLDOUT",json.dumps(result),flush=True)
    pd.DataFrame(out).to_csv("nfl_independent_markets_results.csv",index=False)
    print("NOTE: standalone predictive comparison, NOT an odds-based ROI backtest.",flush=True)
if __name__=="__main__":main()
