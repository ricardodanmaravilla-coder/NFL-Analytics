"""ATS v7: selective disagreement signal. Validation only; 2026 excluded."""
import os, numpy as np, pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss, log_loss
from modules.nfl_calibration import historico_antes
from modules.nfl_ml_engine import PredictorNFL_ML
MIN_P=58.; MIN_EDGE=3.; MIN_EV=3.
def dec(x): x=float(x); return 1+x/100 if x>0 else 1+100/abs(x)
def nv(a,b):
 da,db=dec(a),dec(b); ia,ib=1/da,1/db; s=ia+ib; return ia/s,ib/s
def ordered(d):
 k=[c for c in ['season','week','gameday','gametime','game_id'] if c in d.columns]; return d.sort_values(k,kind='stable').reset_index(drop=True)
def trainframe(games,pbp):
 b=PredictorNFL_ML(); f=b.construir_features_pregame(games,pbp); cols=b._base_feature_names(); pc=b._pbp_feature_names()
 if not pbp.empty and all(c in f.columns for c in pc): cols+=pc
 e=games[['game_id','spread_line','result','home_spread_odds','away_spread_odds']].copy(); f=f.merge(e,on='game_id',how='left')
 f['market_spread_home']=pd.to_numeric(f.spread_line,errors='coerce'); f['ats_margin']=pd.to_numeric(f.result,errors='coerce')-f.market_spread_home
 return b,ordered(f),cols+['market_spread_home']
def xrow(b,pbp,w,h,a,g,line,cols):
 H=b.historial_actual
 if h not in H or a not in H or len(H[h]['pf'])<4 or len(H[a]['pf'])<4:return None
 num=lambda v: pd.to_numeric(v,errors='coerce'); t,wi,hr,ar=num(g.get('temp')),num(g.get('wind')),num(g.get('home_rest')),num(g.get('away_rest'))
 r={'week':float(w),'home_altitude':float(b.altitud_estadios.get(h,0)),'temp':0 if pd.isna(t) else float(t),'wind':0 if pd.isna(wi) else float(wi),'is_dome':int(str(g.get('roof','')).lower() in {'dome','closed','indoor','indoors'}),'temp_missing':int(pd.isna(t)),'wind_missing':int(pd.isna(wi)),'home_rest':0 if pd.isna(hr) else float(hr),'away_rest':0 if pd.isna(ar) else float(ar),'market_spread_home':float(line)}
 r.update(b._features_equipo(H,h,'home')); r.update(b._features_equipo(H,a,'away'))
 if any(c.startswith('home_off_epa_play_') for c in cols):
  from modules.nfl_pbp_engine import features_pbp_actuales
  q=features_pbp_actuales(pbp,h,a)
  if q is None:return None
  r.update(q)
 x=pd.DataFrame([r]); return x[cols] if all(c in x for c in cols) and not x[cols].isna().any(axis=None) else None
def main():
 raw=ordered(nfl.import_schedules([2021,2022,2023,2024,2025])); raw=raw[raw.result.notna()].copy()
 if 'game_type' in raw: raw=raw[raw.game_type.isin(['REG','POST','WC','DIV','CON','SB'])]
 pf='data/historico_nfl_pbp_team_game.csv'; pbp=pd.read_csv(pf) if os.path.exists(pf) else pd.DataFrame(); out=[]
 for s in [2023,2024,2025]:
  for w in sorted(pd.to_numeric(raw[raw.season==s].week,errors='coerce').dropna().astype(int).unique()):
   past=historico_antes(raw,s,w); pp=historico_antes(pbp,s,w) if not pbp.empty else pd.DataFrame(); b,tr,cols=trainframe(past,pp); tr=ordered(tr.dropna(subset=cols+['ats_margin']))
   if len(tr)<300:continue
   # Conservative model predicts only expected ATS residual in points; uncertainty comes only from prior residuals.
   cut=max(250,int(len(tr)*.8)); fit,cal=tr.iloc[:cut],tr.iloc[cut:]
   if len(cal)<40:continue
   m=RandomForestRegressor(n_estimators=500,max_depth=5,min_samples_leaf=25,max_features=.6,random_state=207,n_jobs=1).fit(fit[cols],fit.ats_margin)
   cal_mu=m.predict(cal[cols]); resid=cal.ats_margin.to_numpy()-cal_mu
   m.fit(tr[cols],tr.ats_margin)
   for _,g in ordered(raw[(raw.season==s)&(pd.to_numeric(raw.week,errors='coerce')==w)]).iterrows():
    line=pd.to_numeric(g.spread_line,errors='coerce'); ho=pd.to_numeric(g.get('home_spread_odds'),errors='coerce'); ao=pd.to_numeric(g.get('away_spread_odds'),errors='coerce')
    if pd.isna(line) or pd.isna(ho) or pd.isna(ao) or str(g.get('location','')).lower()=='neutral':continue
    phm,pam=nv(ho,ao); x=xrow(b,pp,w,g.home_team,g.away_team,g,line,cols)
    if x is None:continue
    mu=float(m.predict(x)[0]); ph=float(np.mean(mu+resid>0)); ph=float(np.clip(ph,.01,.99)); cover=float(g.result)-float(line)
    if abs(cover)<1e-9:continue
    y=int(cover>0)
    out.append({'season':s,'week':w,'game_id':g.game_id,'home':g.home_team,'away':g.away_team,'line':float(line),'mu_ats_points':mu,'model_p_home':ph,'market_p_home':phm,'delta_pp':100*(ph-phm),'home_cover':y,'home_odds':float(ho),'away_odds':float(ao)})
 A=pd.DataFrame(out); A.to_csv('backtest_nfl_spread_selective_signal_all.csv',index=False)
 if A.empty:raise SystemExit('No OOS')
 print('=== ALL OOS ===')
 for s,g in A.groupby('season'):
  y=g.home_cover; p=np.clip(g.model_p_home,1e-6,1-1e-6); mp=np.clip(g.market_p_home,1e-6,1-1e-6)
  print(s,{'n':len(g),'brier_model':round(brier_score_loss(y,p),4),'brier_market':round(brier_score_loss(y,mp),4),'logloss_model':round(log_loss(y,p,labels=[0,1]),4),'logloss_market':round(log_loss(y,mp,labels=[0,1]),4)})
 # Fixed economic gates; no post-hoc role/line thresholds.
 picks=[]
 for _,r in A.iterrows():
  for side,p,mp,odd,ln,win in [('H',r.model_p_home,r.market_p_home,r.home_odds,-r.line,r.home_cover),('A',1-r.model_p_home,1-r.market_p_home,r.away_odds,r.line,1-r.home_cover)]:
   prob=100*p; edge=100*(p-mp); ev=(p*dec(odd)-1)*100
   if prob>=MIN_P and edge>=MIN_EDGE and ev>=MIN_EV:
    picks.append({'season':r.season,'game_id':r.game_id,'side':side,'role':'FAVORITE' if ln<0 else 'UNDERDOG','line':ln,'probability':prob,'edge':edge,'ev':ev,'odds':odd,'win':win,'return':dec(odd)-1 if win else -1})
 P=pd.DataFrame(picks); P.to_csv('backtest_nfl_spread_selective_signal_picks.csv',index=False)
 if P.empty: print('NO PICKS'); return
 print('=== PICKS FIXED 58/3/3 ==='); print({'n':len(P),'wr':round(100*P.win.mean(),2),'roi':round(100*P['return'].mean(),2)})
 print(P.groupby(['season','role']).agg(n=('win','size'),wr=('win','mean'),roi=('return','mean'),avg_p=('probability','mean')).to_string())
if __name__=='__main__':main()
