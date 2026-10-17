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
    out={"method":"Expanding-year OOS predictions; historical odds approximate and not execution quotes","seasons":{},"selection":{},"caveats":["Historical odds may differ from quotes available at scan time","Selection evaluated on 2023-24; 2025 is untouched holdout","One season of holdout is insufficient to guarantee profit"]}
    candidates=[1.0,1.5,2.0,2.5]
    evaluations={m:{t:[] for t in candidates} for m in ("SPREAD","TOTAL")}
    holdout={m:{} for m in ("SPREAD","TOTAL")}
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
            if market not in evaluations or year<2023:continue
            # Only residuals from PREVIOUS held-out seasons, never target year.
            errors=[]
            for prior in range(2022,year):
                base=merged[merged.season<prior];validation=merged[merged.season==prior]
                if len(base)<100 or len(validation)<100:continue
                m=IndependentMarketEngine(market,plans[market][variant]).fit(base)
                errors.extend((m.target(validation).to_numpy()-m.predict(validation)).tolist())
            errors=np.asarray(errors,dtype=float)
            errors=errors[np.isfinite(errors)]
            if len(errors)<100:continue
            for temp in candidates:
                picks=[]
                for idx,(_,row) in enumerate(te.iterrows()):
                    if market=="SPREAD":
                        line=row.spread_line
                        if pd.isna(line):continue
                        # NFL schedule spread_line: positive means home favored, hence home handicap = -spread_line.
                        home_p=float(np.mean(pred[idx]+temp*errors>float(line)))
                        away_p=float(np.mean(pred[idx]+temp*errors<float(line)))
                        options=[(home_p,row.home_spread_odds,decimal(row.away_spread_odds),float(target[idx])>float(line)),
                                 (away_p,row.away_spread_odds,decimal(row.home_spread_odds),float(target[idx])<float(line))]
                        push=float(target[idx])==float(line)
                    else:
                        line=row.total_line
                        if pd.isna(line):continue
                        home_p=float(np.mean(pred[idx]+temp*errors>float(line)))
                        away_p=float(np.mean(pred[idx]+temp*errors<float(line)))
                        options=[(home_p,row.over_odds,decimal(row.under_odds),float(target[idx])>float(line)),
                                 (away_p,row.under_odds,decimal(row.over_odds),float(target[idx])<float(line))]
                        push=float(target[idx])==float(line)
                    for prob,odd,other,won in options:
                        d=decimal(odd)
                        if d is None or other is None:continue
                        market_p=(1/d)/((1/d)+(1/other))
                        if prob>=.58 and prob-market_p>=.03 and prob*d-1>=.03:
                            picks.append(0.0 if push else d-1 if won else -1.0)
                result={"n":len(picks),"roi":round(float(np.mean(picks)),4) if picks else None,
                        "wins":sum(x>0 for x in picks),"losses":sum(x<0 for x in picks)}
                (holdout if year==2025 else evaluations)[market].setdefault(temp,[]).append(result)
        out["seasons"][str(year)]=yr
        print(json.dumps({"season":year,"results":yr}),flush=True)
    for market in evaluations:
        rows={}
        for temp,years in evaluations[market].items():
            total=sum(x["n"] for x in years)
            roi=sum(x["n"]*x["roi"] for x in years if x["roi"] is not None)/total if total else None
            rows[str(temp)]={"n":total,"roi":round(roi,4) if roi is not None else None}
        eligible=[(float(t),x) for t,x in rows.items() if x["n"]>=25 and x["roi"] is not None]
        chosen=max(eligible,key=lambda item:item[1]["roi"])[0] if eligible else None
        out["selection"][market]={"training_2023_2024":rows,"candidate":chosen,
            "holdout_2025":holdout[market].get(chosen,[]) if chosen is not None else [],
            "baseline_2x_holdout_2025":holdout[market].get(2.0,[])}
    print("CALIBRATION",json.dumps(out["selection"]),flush=True)
    Path("data/nfl_shadow").mkdir(parents=True,exist_ok=True)
    Path("data/nfl_shadow/independent_backtest_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()
