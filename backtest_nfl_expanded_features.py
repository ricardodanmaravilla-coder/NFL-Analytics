"""Offline-only expanded-feature experiment; production remains unchanged.
Season holdouts compare identical games. No betting is authorized by this script.
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

def main():
    data=Path("data")
    g=pd.read_csv(data/"historico_nfl_games.csv")
    p=pd.read_csv(data/"historico_nfl_pbp_team_game.csv")
    q=pd.read_csv(data/"historico_nfl_qbs.csv")
    g=g[pd.to_numeric(g.season,errors="coerce")<=2025]
    p=p[pd.to_numeric(p.season,errors="coerce")<=2025]
    q=q[pd.to_numeric(q.season,errors="coerce")<=2025]
    f=build_expanded_pregame(g,p,q)
    base=PredictorNFL_ML()._base_feature_names()
    reserved={"game_id","season","week","home_team","away_team","puntos_totales","margen_local"}
    enriched=[c for c in f.columns if c not in reserved and pd.api.types.is_numeric_dtype(f[c])]
    enriched=sorted(set(enriched)|set(base))
    results=[]
    for year in (2023,2024,2025):
        train=f[(f.season<year)&(f.season>=2021)]
        test=f[f.season==year]
        if len(train)<150 or len(test)<50:continue
        for target in ("margen_local","puntos_totales"):
            for variant,cols in (("BASELINE",base),("EXPANDED",enriched)):
                xtr=train[cols].replace([np.inf,-np.inf],np.nan)
                xte=test[cols].replace([np.inf,-np.inf],np.nan)
                # Train-only imputation prevents target-season distribution leakage.
                usable=[c for c in cols if xtr[c].notna().sum()>=20]
                imp=SimpleImputer(strategy="median",keep_empty_features=True)
                a=imp.fit_transform(xtr[usable]);b=imp.transform(xte[usable])
                model=RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,
                                            random_state=42 if target=="puntos_totales" else 43,n_jobs=2)
                model.fit(a,train[target]);pred=model.predict(b)
                actual=test[target].to_numpy(dtype=float)
                row={"year":year,"target":target,"variant":variant,"n":len(test),
                     "features":len(usable),"mae":round(float(mean_absolute_error(actual,pred)),4),
                     "rmse":round(float(np.sqrt(mean_squared_error(actual,pred))),4)}
                if target=="margen_local":
                    row["winner_accuracy"]=round(float(np.mean((pred>0)==(actual>0))),4)
                results.append(row)
                print(json.dumps(row),flush=True)
    if not results:raise SystemExit("No hay muestra suficiente para validación")
    out=pd.DataFrame(results)
    out.to_csv("nfl_expanded_feature_comparison.csv",index=False)
    print("\nPAIRWISE",flush=True)
    for (year,target),part in out.groupby(["year","target"]):
        if set(part.variant)=={"BASELINE","EXPANDED"}:
            b=part.set_index("variant")
            print(json.dumps({"year":int(year),"target":target,
                "mae_improvement":round(float(b.loc["BASELINE","mae"]-b.loc["EXPANDED","mae"]),4),
                "baseline_features":int(b.loc["BASELINE","features"]),
                "expanded_features":int(b.loc["EXPANDED","features"])}),flush=True)

if __name__=="__main__":main()
