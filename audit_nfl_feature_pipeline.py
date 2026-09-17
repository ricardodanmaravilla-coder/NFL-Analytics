"""Read-only audit of Big Data -> pregame features. 2026 excluded from validation."""
from pathlib import Path
import json, numpy as np, pandas as pd
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_pbp_engine import construir_pbp_pregame, PBP_METRICS
D=Path('data'); O={}
def add(k,**v): O[k]=v; print(k,v)
def main():
 g=pd.read_csv(D/'historico_nfl_games.csv'); p=pd.read_csv(D/'historico_nfl_pbp_team_game.csv')
 g['season']=pd.to_numeric(g.season,errors='coerce'); g['week']=pd.to_numeric(g.week,errors='coerce')
 p['season']=pd.to_numeric(p.season,errors='coerce'); p['week']=pd.to_numeric(p.week,errors='coerce')
 histg=g[g.season<=2025].copy(); histp=p[p.season<=2025].copy()
 m=PredictorNFL_ML(); f=m.construir_features_pregame(histg,histp)
 add('feature_frame',rows=len(f),seasons=sorted(pd.to_numeric(f.season,errors='coerce').dropna().astype(int).unique().tolist()),columns=len(f.columns),duplicate_game_rows=int(f.duplicated(['game_id'],keep=False).sum()))
 # Independently recompute score rolling features from strictly prior weeks and compare exact values.
 expected={}; history={}
 for (s,w),wk in histg.sort_values(['season','week','game_id']).groupby(['season','week'],sort=True):
  for _,r in wk.iterrows():
   for side,team in [('home',r.home_team),('away',r.away_team)]:
    h=history.get(team,[])
    if len(h)>=4:
     expected[(str(r.game_id),side,'off_5')]=float(np.mean([x[0] for x in h[-5:]]))
     expected[(str(r.game_id),side,'def_5')]=float(np.mean([x[1] for x in h[-5:]]))
     expected[(str(r.game_id),side,'margin_5')]=float(np.mean([x[0]-x[1] for x in h[-5:]]))
  for _,r in wk.iterrows():
   if pd.notna(r.home_score) and pd.notna(r.away_score):
    history.setdefault(r.home_team,[]).append((float(r.home_score),float(r.away_score)))
    history.setdefault(r.away_team,[]).append((float(r.away_score),float(r.home_score)))
 errs=[]; checked=0
 for _,r in f.iterrows():
  for side in ['home','away']:
   for metric in ['off_5','def_5','margin_5']:
    key=(str(r.game_id),side,metric)
    if key in expected:
     checked+=1; errs.append(abs(float(r[f'{side}_{metric}'])-expected[key]))
 add('score_feature_recompute',checked=checked,mismatches=int(sum(e>1e-9 for e in errs)),max_abs=float(max(errs) if errs else 0))
 # PBP feature chronology: reconstruct source games and ensure every contributing game is from a strictly earlier week tuple.
 pp=construir_pbp_pregame(histg,histp); merged=f[['game_id','season','week']].merge(pp,on='game_id',how='left',indicator=True)
 add('pbp_feature_join',feature_games=len(f),matched=int((merged._merge=='both').sum()),missing=int((merged._merge!='both').sum()))
 # Exact independent PBP rolling recomputation for all model PBP columns.
 history={}; exp_rows=[]
 game_min=histg[['game_id','season','week','home_team','away_team']].sort_values(['season','week','game_id'])
 for (s,w),wk in game_min.groupby(['season','week'],sort=True):
  for _,r in wk.iterrows():
   row={'game_id':r.game_id}; ok=True
   for side,team in [('home',r.home_team),('away',r.away_team)]:
    h=history.get(team,[])
    if len(h)<4: ok=False; break
    hd=pd.DataFrame(h)
    for metric in PBP_METRICS:
     for win in [4,8]: row[f'{side}_{metric}_{win}']=pd.to_numeric(hd.get(metric),errors='coerce').tail(win).mean()
   if ok: exp_rows.append(row)
  ids=set(wk.game_id.astype(str)); wp=histp[histp.game_id.astype(str).isin(ids)]
  for _,r in wp.iterrows():
   history.setdefault(r.team,[]).append({x:r.get(x) for x in PBP_METRICS}); history[r.team]=history[r.team][-16:]
 exp=pd.DataFrame(exp_rows); cols=[c for c in exp.columns if c!='game_id']; z=pp.merge(exp,on='game_id',suffixes=('_actual','_expected'))
 diffs=[]
 for c in cols:
  a=pd.to_numeric(z[c+'_actual'],errors='coerce'); b=pd.to_numeric(z[c+'_expected'],errors='coerce'); mask=a.notna()&b.notna(); diffs.extend((a[mask]-b[mask]).abs().tolist())
 add('pbp_feature_recompute',games=len(z),values=len(diffs),mismatches=int(sum(x>1e-9 for x in diffs)),max_abs=float(max(diffs) if diffs else 0))
 # Current prediction path leakage risk: features_pbp_actuales has no as-of cutoff. Detect whether 2026 PBP exists and therefore can enter a historical/as-of prediction if caller does not prefilter.
 future= p[p.season>=2026]
 add('current_pbp_asof_risk',rows_2026=len(future),teams_2026=int(future.team.nunique()) if len(future) else 0,warning='features_pbp_actuales uses latest rows with no explicit as-of cutoff')
 # Verify official nflverse spread convention against internal score result formula.
 r=pd.to_numeric(histg.result,errors='coerce'); s=pd.to_numeric(histg.spread_line,errors='coerce'); valid=r.notna()&s.notna();
 add('spread_target_semantics',n=int(valid.sum()),home_cover_rate_result_minus_spread=float(((r-s)>0)[valid].mean()),pushes=int(((r-s).abs()<1e-9)[valid].sum()),definition='nflverse: positive spread_line means home favored; home ATS margin = result - spread_line')
 Path('audit_nfl_feature_pipeline.json').write_text(json.dumps(O,indent=2,default=str),encoding='utf-8')
 print('FEATURE_PIPELINE_AUDIT_COMPLETE')
if __name__=='__main__': main()
