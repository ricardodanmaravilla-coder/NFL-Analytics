"""ATS v6 validation: market-first guard. No ML probability fabrication; 2026 excluded."""
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.metrics import brier_score_loss, log_loss

def dec(x):
 x=float(x); return 1+x/100 if x>0 else 1+100/abs(x)
def nv(a,b):
 da,db=dec(a),dec(b); ia,ib=1/da,1/db; s=ia+ib; return ia/s,ib/s

def main():
 d=nfl.import_schedules([2023,2024,2025]); d=d[d.result.notna()].copy()
 if 'game_type' in d: d=d[d.game_type.isin(['REG','POST','WC','DIV','CON','SB'])]
 rows=[]
 for _,g in d.iterrows():
  line=pd.to_numeric(g.get('spread_line'),errors='coerce'); ho=pd.to_numeric(g.get('home_spread_odds'),errors='coerce'); ao=pd.to_numeric(g.get('away_spread_odds'),errors='coerce')
  if pd.isna(line) or pd.isna(ho) or pd.isna(ao) or str(g.get('location','')).lower()=='neutral': continue
  cover=float(g.result)-float(line)
  if abs(cover)<1e-9: continue
  ph,pa=nv(ho,ao); y=int(cover>0)
  rows.append({'season':int(g.season),'week':int(g.week),'game_id':g.game_id,'home':g.home_team,'away':g.away_team,'spread_line':float(line),'market_p_home':ph,'home_cover':y,'home_odds':float(ho),'away_odds':float(ao)})
 A=pd.DataFrame(rows)
 print('=== MARKET ATS BASELINE OOS ===')
 for s,g in A.groupby('season'):
  y=g.home_cover.to_numpy(); p=g.market_p_home.to_numpy(); print(s,{'n':len(g),'home_cover':round(y.mean(),4),'brier':round(brier_score_loss(y,p),4),'logloss':round(log_loss(y,p,labels=[0,1]),4)})
 print('\n=== MARKET ROLE / LINE AUDIT (diagnostic only; no threshold tuning) ===')
 z=[]
 for _,r in A.iterrows():
  for side,p,odd,line,win in [('H',r.market_p_home,r.home_odds,-r.spread_line,r.home_cover),('A',1-r.market_p_home,r.away_odds,r.spread_line,1-r.home_cover)]:
   role='FAVORITE' if line<0 else ('UNDERDOG' if line>0 else 'PICKEM')
   z.append({'season':r.season,'role':role,'line':line,'market_p':100*p,'win':win,'return':dec(odd)-1 if win else -1})
 Z=pd.DataFrame(z)
 print(Z.groupby(['season','role']).agg(n=('win','size'),wr=('win','mean'),roi=('return','mean'),avg_market_p=('market_p','mean')).to_string())
 print('\nSafety conclusion input: market baseline is the fallback when learned ATS corrections fail calibration.')
 A.to_csv('backtest_nfl_spread_market_guard_all.csv',index=False)
if __name__=='__main__': main()
