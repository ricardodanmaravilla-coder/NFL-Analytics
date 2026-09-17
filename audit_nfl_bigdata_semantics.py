"""Second-stage read-only audit: QB coverage, market semantics, temporal leakage, feature sanity."""
from pathlib import Path
import json, pandas as pd, numpy as np
D=Path('data'); O={}
def add(k,**v): O[k]=v; print(k,v)
def n(s): return pd.to_numeric(s,errors='coerce')
def main():
 g=pd.read_csv(D/'historico_nfl_games.csv'); p=pd.read_csv(D/'historico_nfl_pbp_team_game.csv'); q=pd.read_csv(D/'historico_nfl_qbs.csv')
 add('qb_schema',columns=q.columns.tolist(),rows=len(q))
 if 'season' in q: add('qb_seasons',counts={str(k):int(v) for k,v in q.groupby(n(q.season)).size().items()})
 for c in ['player_name','passing_yards','season','week','team','opponent']:
  if c in q: add('qb_'+c,nulls=int(q[c].isna().sum()),null_rate=float(q[c].isna().mean()),unique=int(q[c].nunique(dropna=True)))
 if {'player_name','season','week'}<=set(q): add('qb_duplicate_player_week',rows=int(q.duplicated(['player_name','season','week'],keep=False).sum()))
 if 'passing_yards' in q:
  x=n(q.passing_yards); add('qb_passing_yards',negative=int((x<0).fillna(False).sum()),min=float(x.min()),median=float(x.median()),max=float(x.max()))
 # Market semantics: test which spread convention matches score margin and favorite prices.
 req={'result','spread_line','home_moneyline','away_moneyline'}
 if req<=set(g):
  z=g[list(req)].apply(n).dropna(); sp=z.spread_line
  # If spread_line is home-team line, negative should usually pair with shorter home ML. If favorite-margin convention, positive should.
  home_short=z.home_moneyline < z.away_moneyline
  add('spread_semantics',rows=len(z),corr_spread_result=float(sp.corr(z.result)),neg_line_home_short=float(home_short[sp<0].mean()) if (sp<0).any() else None,pos_line_home_short=float(home_short[sp>0].mean()) if (sp>0).any() else None,zero_lines=int((sp==0).sum()))
 if {'result','spread_line'}<=set(g):
  r=n(g.result); s=n(g.spread_line); valid=r.notna()&s.notna()
  # Report both candidate ATS identities; this does not assume one is correct.
  add('ats_candidate_outcomes',n=int(valid.sum()),home_cover_if_result_plus_spread=float(((r+s)>0)[valid].mean()),home_cover_if_result_minus_spread=float(((r-s)>0)[valid].mean()),push_plus=int(((r+s).abs()<1e-9)[valid].sum()),push_minus=int(((r-s).abs()<1e-9)[valid].sum()))
 # Temporal leakage: every team-game PBP row must be unique in season/week and history strictly prior when rolling.
 if {'team','season','week','game_id'}<=set(p):
  x=p.copy(); x['_s']=n(x.season); x['_w']=n(x.week); x=x.sort_values(['team','_s','_w','game_id'])
  same_week=x.duplicated(['team','_s','_w'],keep=False)
  add('pbp_team_week',duplicate_team_week_rows=int(same_week.sum()))
 # Feature source columns that are suspiciously outcome/market-derived.
 outcome_tokens=('score','result','winner','cover','ats','final','margin')
 suspicious=[c for c in p.columns if any(t in c.lower() for t in outcome_tokens)]
 add('pbp_outcome_like_columns',columns=suspicious)
 # Games schedule chronology and duplicate team appearances in same week.
 if {'season','week','home_team','away_team'}<=set(g):
  tg=pd.concat([g[['season','week','home_team']].rename(columns={'home_team':'team'}),g[['season','week','away_team']].rename(columns={'away_team':'team'})])
  add('games_team_week',duplicate_team_week_rows=int(tg.duplicated(['season','week','team'],keep=False).sum()))
 # Missing market lines by season, crucial to backtests.
 for c in ['spread_line','total_line','home_moneyline','away_moneyline','home_spread_odds','away_spread_odds']:
  if c in g:
   rates={str(int(k)):round(float(v),6) for k,v in g.assign(_miss=g[c].isna()).groupby('season')._miss.mean().items()}
   add('market_missing_'+c,rates=rates)
 Path('audit_nfl_bigdata_semantics.json').write_text(json.dumps(O,indent=2,default=str),encoding='utf-8')
 print('SEMANTIC_AUDIT_COMPLETE')
if __name__=='__main__': main()
