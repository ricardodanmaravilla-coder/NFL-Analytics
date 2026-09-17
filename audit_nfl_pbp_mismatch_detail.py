"""Read-only drilldown for PBP pregame mismatch root cause. 2026 excluded."""
import pandas as pd, json
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_bigdata_store import cargar_pbp_preferente

def main():
    g=pd.read_csv('data/historico_nfl_games.csv')
    g['season']=pd.to_numeric(g['season'],errors='coerce')
    g=g[g['season']<=2025].copy()
    p=cargar_pbp_preferente(seasons=[2021,2022,2023,2024,2025]).copy()
    m=PredictorNFL_ML(); f=m.construir_features_pregame(g,p)
    out={'feature_rows':len(f),'pbp_rows':len(p)}
    tokens=('epa','success','explosive','sack','pressure','plays')
    cols=[c for c in f.columns if any(t in c.lower() for t in tokens)]
    out['pbp_feature_columns']=cols
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
            # rolling feature suffix is _4/_8; compare only when source metric exists
            base=c[len(prefix):]
            if base.endswith('_4') or base.endswith('_8'): base=base.rsplit('_',1)[0]
            if base not in pr.index: continue
            a=pd.to_numeric(pd.Series([r[c]]),errors='coerce').iloc[0]; b=pd.to_numeric(pd.Series([pr[base]]),errors='coerce').iloc[0]
            if pd.notna(a) and pd.notna(b):
              checked+=1
              if abs(float(a)-float(b))<1e-10: leaks.append((gid,side,team,c,float(a)))
    out['same_game_exact_matches']={'checked':checked,'matches':len(leaks),'rate':len(leaks)/checked if checked else None,'examples':leaks[:20]}
    # Current helper risk is assessed separately: full store may contain future rows; historical constructor above is bounded to <=2025.
    full=cargar_pbp_preferente().copy(); full['season']=pd.to_numeric(full['season'],errors='coerce')
    future=full[full['season']>=2026]
    out['unbounded_current_helper_risk']={'rows_2026':len(future),'teams_2026':int(future['team'].nunique()) if len(future) else 0}
    open('audit_nfl_pbp_mismatch_detail.json','w').write(json.dumps(out,indent=2,default=str))
    print(json.dumps(out,indent=2,default=str)); print('PBP_MISMATCH_DETAIL_COMPLETE')
if __name__=='__main__': main()
