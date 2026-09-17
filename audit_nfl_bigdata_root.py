"""Read-only root audit of NFL Big Data. Does not alter production or stored datasets."""
from pathlib import Path
import json, numpy as np, pandas as pd
from modules.nfl_bigdata_store import cargar_pbp_preferente, validar_paridad_csv_parquet

DATA=Path('data'); OUT={}
def add(k,**v): OUT[k]=v; print(k,v)
def dup(df,cols): return int(df.duplicated(cols,keep=False).sum()) if set(cols)<=set(df.columns) else -1
def num(s): return pd.to_numeric(s,errors='coerce')
def main():
 games=pd.read_csv(DATA/'historico_nfl_games.csv'); pbp=pd.read_csv(DATA/'historico_nfl_pbp_team_game.csv'); qbs=pd.read_csv(DATA/'historico_nfl_qbs.csv')
 add('files',games_rows=len(games),pbp_rows=len(pbp),qb_rows=len(qbs),games_cols=len(games.columns),pbp_cols=len(pbp.columns),qb_cols=len(qbs.columns))
 # schemas, seasons, keys
 for name,d,key in [('games',games,['game_id']),('pbp',pbp,['game_id','team']),('qbs',qbs,['game_id','player_id'])]:
  seasons=sorted(num(d.season).dropna().astype(int).unique().tolist()) if 'season' in d else []
  add(name+'_keys',seasons=seasons,duplicate_key_rows=dup(d,key),null_game_id=int(d.game_id.isna().sum()) if 'game_id' in d else -1)
 # games structural truth
 req=['game_id','season','week','home_team','away_team','home_score','away_score','result','spread_line','total_line']
 add('games_schema',missing=[c for c in req if c not in games])
 if {'home_score','away_score','result'}<=set(games):
  hs,as_,r=num(games.home_score),num(games.away_score),num(games.result); m=hs.notna()&as_.notna()&r.notna()
  add('games_result_identity',checked=int(m.sum()),mismatch=int(((hs-as_-r).abs()>1e-9)[m].sum()),max_abs=float((hs-as_-r)[m].abs().max()) if m.any() else None)
 add('games_teams',same_home_away=int((games.home_team==games.away_team).sum()) if {'home_team','away_team'}<=set(games) else -1)
 # PBP: exactly two team rows per game expected; opponents reciprocal
 cnt=pbp.groupby('game_id').team.nunique(); add('pbp_game_coverage',games=int(cnt.size),not_two_teams=int((cnt!=2).sum()),min_teams=int(cnt.min()),max_teams=int(cnt.max()))
 if 'opponent' in pbp:
  pairs=pbp[['game_id','team','opponent']].dropna(); rev=pairs.rename(columns={'team':'opponent','opponent':'team'}); chk=pairs.merge(rev,on=['game_id','team','opponent'],how='left',indicator=True)
  add('pbp_reciprocity',rows=len(pairs),missing_reverse=int((chk._merge!='both').sum()),self_opponent=int((pairs.team==pairs.opponent).sum()))
 # Metric domains and missingness
 domains={
  'off_success_rate':(0,1),'explosive_rate':(0,1),'sack_rate_allowed':(0,1),'def_success_allowed':(0,1),'def_explosive_allowed':(0,1),'pressure_rate':(0,1),
  'neutral_pass_rate':(0,1),'early_down_success':(0,1),'redzone_success':(0,1),'late_down_success':(0,1)}
 bad={}; missing={}
 for c in pbp.columns:
  x=num(pbp[c]);
  if x.notna().any(): missing[c]=round(float(x.isna().mean()),6)
 for c,(lo,hi) in domains.items():
  if c in pbp:
   x=num(pbp[c]); bad[c]=int(((x<lo)|(x>hi)).fillna(False).sum())
 add('pbp_domains',out_of_range=bad)
 add('pbp_missingness',worst=sorted(missing.items(),key=lambda z:z[1],reverse=True)[:15])
 if 'plays' in pbp:
  x=num(pbp.plays); add('pbp_plays',nonpositive=int((x<=0).fillna(False).sum()),min=float(x.min()),median=float(x.median()),max=float(x.max()))
 # join coverage games <-> PBP and team identity
 gteams=pd.concat([games[['game_id','home_team']].rename(columns={'home_team':'team'}),games[['game_id','away_team']].rename(columns={'away_team':'team'})],ignore_index=True).dropna()
 pkeys=pbp[['game_id','team']].dropna().drop_duplicates(); j=gteams.merge(pkeys,on=['game_id','team'],how='left',indicator=True)
 add('games_pbp_join',expected_team_games=len(gteams),missing_pbp_team_games=int((j._merge!='both').sum()),coverage=round(float((j._merge=='both').mean()),6))
 # temporal sanity: PBP season/week should agree with games
 if {'season','week'}<=set(games) and {'season','week'}<=set(pbp):
  z=pbp[['game_id','season','week']].merge(games[['game_id','season','week']],on='game_id',suffixes=('_p','_g'),how='inner'); add('temporal_alignment',rows=len(z),season_mismatch=int((num(z.season_p)!=num(z.season_g)).sum()),week_mismatch=int((num(z.week_p)!=num(z.week_g)).sum()))
 # CSV/parquet parity is critical
 try: add('csv_parquet_parity',**validar_paridad_csv_parquet())
 except Exception as e: add('csv_parquet_parity',ok=False,error=repr(e))
 # QB basic duplicate/coverage diagnostics
 if {'game_id','team'}<=set(qbs):
  qc=qbs.groupby(['game_id','team']).size(); add('qb_team_game',groups=len(qc),multirow_groups=int((qc>1).sum()),max_rows=int(qc.max()))
 # cross-season leakage indicators: duplicate game ids across seasons
 for name,d in [('games',games),('pbp',pbp),('qbs',qbs)]:
  if {'game_id','season'}<=set(d):
   n=d.groupby('game_id').season.nunique(); add(name+'_crossseason_gameid',ids_multi_season=int((n>1).sum()))
 Path('audit_nfl_bigdata_root.json').write_text(json.dumps(OUT,indent=2,default=str),encoding='utf-8')
 print('AUDIT_COMPLETE')
if __name__=='__main__': main()
