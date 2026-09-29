"""NFL feature-family ablation: select on 2023-24; untouched 2025 final check.
Research only. No changes to production model, odds, or auto-betting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules.nfl_experimental_features import build_expanded_pregame
from modules.nfl_ml_engine import PredictorNFL_ML

RESERVED={"game_id","season","week","home_team","away_team","puntos_totales","margen_local"}
def families(f,base):
    numeric=[c for c in f if c not in RESERVED and pd.api.types.is_numeric_dtype(f[c])]
    other=[c for c in numeric if c not in base]
    groups={
      "qb_efficiency":[c for c in other if "_qb_" in c and "vs_opponent" not in c],
      "qb_matchup":[c for c in other if "_qb_vs_opponent_" in c],
      "team_pbp":[c for c in other if any(c.startswith(s+"_") for s in ("home","away")) and any("_"+m+"_" in c for m in ("off_epa_play","off_success_rate","pass_epa","rush_epa","explosive_rate","sack_rate_allowed","plays","def_epa_allowed","def_success_allowed","def_explosive_allowed","pressure_rate"))],
      "situational_pbp":[c for c in other if any("_"+m+"_" in c for m in ("early_down_epa","early_down_success","neutral_pass_rate","redzone_epa","redzone_success","third_fourth_epa","late_down_success","early_down_epa_allowed","redzone_epa_allowed","late_down_epa_allowed"))],
      "venue":[c for c in other if "_venue_" in c],
      "team_matchup":[c for c in other if "_vs_opponent_" in c and "_qb_" not in c],
      "kickoff":[c for c in other if c in ("night_game","night_missing","neutral_site")],
    }
    covered=set(sum(groups.values(),[]))
    groups["other"]=[c for c in other if c not in covered]
    return {k:sorted(set(v)) for k,v in groups.items() if v}

def fit_predict(train,test,cols,target):
    x=train[cols].replace([np.inf,-np.inf],np.nan)
    xt=test[cols].replace([np.inf,-np.inf],np.nan)
    usable=[c for c in cols if x[c].notna().sum()>=20]
    if not usable:raise ValueError("No usable features")
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    a=imputer.fit_transform(x[usable]);b=imputer.transform(xt[usable])
    model=RandomForestRegressor(n_estimators=160,max_depth=9,min_samples_leaf=6,
        random_state=43 if target=="margen_local" else 42,n_jobs=2)
    model.fit(a,train[target])
    pred=model.predict(b);actual=test[target].to_numpy(dtype=float)
    result={"mae":float(mean_absolute_error(actual,pred)),
            "rmse":float(np.sqrt(mean_squared_error(actual,pred))),
            "features":len(usable),"n":len(test)}
    if target=="margen_local":
        result["winner_accuracy"]=float(np.mean((pred>0)==(actual>0)))
    return result

def main():
    root=Path("data")
    games=pd.read_csv(root/"historico_nfl_games.csv")
    pbp=pd.read_csv(root/"historico_nfl_pbp_team_game.csv")
    qbs=pd.read_csv(root/"historico_nfl_qbs.csv")
    games=games[pd.to_numeric(games.season,errors="coerce")<=2025]
    pbp=pbp[pd.to_numeric(pbp.season,errors="coerce")<=2025]
    qbs=qbs[pd.to_numeric(qbs.season,errors="coerce")<=2025]
    f=build_expanded_pregame(games,pbp,qbs)
    base=PredictorNFL_ML()._base_feature_names()
    groups=families(f,base)
    print("FAMILY_COUNTS",json.dumps({k:len(v) for k,v in groups.items()}),flush=True)
    if not groups:raise SystemExit("No expanded feature groups")
    candidates={"BASELINE":base,"ALL":sorted(set(base+sum(groups.values(),[])))}
    for name,cols in groups.items():
        candidates["BASE+"+name]=sorted(set(base+cols))
    # Controlled, predeclared combinations; no 2025 outcome used to select.
    for names in (("qb_efficiency","team_pbp"),("qb_efficiency","situational_pbp"),
                  ("team_pbp","situational_pbp"),("qb_efficiency","venue"),
                  ("qb_efficiency","team_pbp","situational_pbp")):
        if all(x in groups for x in names):
            candidates["BASE+"+"+".join(names)]=sorted(set(base+sum((groups[x] for x in names),[])))
    results=[]
    for target in ("margen_local","puntos_totales"):
        validation={}
        for name,cols in candidates.items():
            metrics=[]
            for year in (2023,2024):
                train=f[(f.season<year)&(f.season>=2021)]
                test=f[f.season==year]
                if len(train)<150 or len(test)<50:raise SystemExit("Insufficient temporal training/validation coverage")
                result=fit_predict(train,test,cols,target)
                result.update({"year":year,"target":target,"variant":name,"split":"SELECTION"})
                results.append(result);metrics.append(result)
                print("VALIDATION",json.dumps(result),flush=True)
            validation[name]=float(np.mean([x["mae"] for x in metrics]))
        # Penalize unnecessary complexity; require >=0.10 point MAE improvement
        # on BOTH selection years, otherwise retain BASELINE.
        baseline=[x for x in results if x["target"]==target and x["variant"]=="BASELINE" and x["split"]=="SELECTION"]
        eligible=[]
        for name in candidates:
            if name=="BASELINE":continue
            trial=[x for x in results if x["target"]==target and x["variant"]==name and x["split"]=="SELECTION"]
            if all(b["mae"]-t["mae"]>=0.10 for b,t in zip(baseline,trial)):
                eligible.append(name)
        chosen=min(eligible,key=lambda x:(validation[x],len(candidates[x]))) if eligible else "BASELINE"
        print("SELECTED",json.dumps({"target":target,"variant":chosen,"selection_mae":validation[chosen],
                                     "rule":"improve MAE >=0.10 on each of 2023 and 2024"}),flush=True)
        # 2025 held out entirely until the selection has been frozen.
        train=f[(f.season<2025)&(f.season>=2021)];test=f[f.season==2025]
        for name in sorted(set(["BASELINE","ALL",chosen])):
            r=fit_predict(train,test,candidates[name],target)
            r.update({"year":2025,"target":target,"variant":name,"split":"LOCKED_HOLDOUT"})
            results.append(r);print("HOLDOUT",json.dumps(r),flush=True)
    pd.DataFrame(results).to_csv("nfl_feature_ablation_results.csv",index=False)
    print("NOTE: schedule weather in historical games may reflect observed conditions, not timestamped forecasts.",flush=True)

if __name__=="__main__":main()
