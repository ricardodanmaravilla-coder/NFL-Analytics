"""Read-only drilldown for PBP pregame mismatch root cause."""
import pandas as pd, numpy as np, json
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_bigdata_store import cargar_pbp_team_game

def main():
    m=PredictorNFL_ML(); f=m.construir_features_pregame(); p=cargar_pbp_team_game().copy()
    out={'feature_rows':len(f),'pbp_rows':len(p)}
    # identify PBP-derived columns in feature frame
    tokens=('epa','success','explosive','sack','pressure','neutral','early','redzone','late','plays')
    cols=[c for c in f.columns if any(t in c.lower() for t in tokens)]
    out['pbp_feature_columns']=cols
    # Check whether feature values for a game equal same-game PBP aggregates: that would be direct leakage.
    leaks=[]; checked=0
    if {'game_id','home_team','away_team'}<=set(f) and {'game_id','team'}<=set(p):
      pi=p.set_index(['game_id','team'])
      for _,r in f.iterrows():
        gid=r.game_id
        for side,team in [('home',r.home_team),('away',r.away_team)]:
          if (gid,team) not in pi.index: continue
          pr=pi.loc[(gid,team)]
          if isinstance(pr,pd.DataFrame): pr=pr.iloc[0]
          for c in cols:
            prefix=side+'_'
            if not c.startswith(prefix): continue
            base=c[len(prefix):]
            if base not in pr.index: continue
            a=pd.to_numeric(pd.Series([r[c]]),errors='coerce').iloc[0]; b=pd.to_numeric(pd.Series([pr[base]]),errors='coerce').iloc[0]
            if pd.notna(a) and pd.notna(b):
              checked+=1
              if abs(float(a)-float(b))<1e-10: leaks.append((gid,side,team,c,float(a)))
    out['same_game_exact_matches']={'checked':checked,'matches':len(leaks),'rate':len(leaks)/checked if checked else None,'examples':leaks[:20]}
    # Temporal source sanity: for each target game, count PBP rows for target teams that are same/future week.
    bad=[]
    if {'season','week','home_team','away_team','game_id'}<=set(f) and {'season','week','team','game_id'}<=set(p):
      p['season']=pd.to_numeric(p.season,errors='coerce'); p['week']=pd.to_numeric(p.week,errors='coerce')
      for _,r in f.iterrows():
        s=float(r.season); w=float(r.week)
        for side,team in [('home',r.home_team),('away',r.away_team)]:
          x=p[(p.team==team)&((p.season>s)|((p.season==s)&(p.week>=w)))]
          # presence is normal in full store; risk only if current helper has no as-of cutoff, so report counts/examples.
          if len(x): bad.append((r.game_id,side,team,int(len(x))))
    out['future_rows_available_to_unbounded_helper']={'target_team_sides':len(bad),'examples':bad[:20]}
    open('audit_nfl_pbp_mismatch_detail.json','w').write(json.dumps(out,indent=2,default=str))
    print(json.dumps(out,indent=2,default=str)); print('PBP_MISMATCH_DETAIL_COMPLETE')
if __name__=='__main__': main()
