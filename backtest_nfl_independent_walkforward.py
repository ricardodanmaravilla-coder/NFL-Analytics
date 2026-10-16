"""Walk-forward audit of NFL Independent. Historical odds are used ONLY for evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_independent_markets import IndependentMarketEngine, feature_groups

ROOT=Path("data")
def decimal(o):
    if pd.isna(o): return None
    o=float(o)
    return 1+o/100 if o>=100 else (1+100/abs(o) if o<=-100 else None)
def score(p,odd):
    d=decimal(odd)
    if d is None:return None
    implied=1/d
    return {"ev":p*d-1,"edge":p-implied}
def main():
    g=pd.read_csv(ROOT/"historico_nfl_games.csv")
    pbp=pd.read_csv(ROOT/"historico_nfl_pbp_team_game.csv")
    qb=pd.read_csv(ROOT/"historico_nfl_qbs.csv")
    g=g.dropna(subset=["home_score","away_score"])
    g=g[g.season.between(2021,2025)].copy()
    f=build_expanded_pregame(g,pbp,qb)
    plans=feature_groups(f)
    merged=f.merge(g[["game_id","spread_line","total_line","home_moneyline","away_moneyline","home_spread_odds","away_spread_odds","over_odds","under_odds"]],on="game_id",how="left",validate="one_to_one")
    out={"method":"Expanding season holdout; train prior seasons only; market odds for grading only","seasons":{},"caveats":["Historical closing odds may not match pregame scan timing","No simulated bankroll compounding","Historical feature availability must be checked independently"]}
    for year in (2022,2023,2024,2025):
        tr=merged[merged.season<year];te=merged[merged.season==year]
        if len(tr)<100 or len(te)<100:continue
        yr={}
        for market,variant in [("ML","FULL"),("SPREAD","BASELINE"),("TOTAL","BASELINE")]:
            model=IndependentMarketEngine(market,plans[market][variant]).fit(tr)
            pred=model.predict(te)
            target=model.target(te).to_numpy()
            stats={"n":len(te),"features":len(model.usable)}
            if market=="ML":
                stats.update({"brier":round(float(brier_score_loss(target,pred)),5),"log_loss":round(float(log_loss(target,np.clip(pred,1e-6,1-1e-6),labels=[0,1])),5)})
            else:stats["mae"]=round(float(mean_absolute_error(target,pred)),3)
            yr[market]=stats
        out["seasons"][str(year)]=yr
        print(json.dumps({"season":year,"results":yr}),flush=True)
    Path("data/nfl_shadow").mkdir(parents=True,exist_ok=True)
    Path("data/nfl_shadow/independent_backtest_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()
