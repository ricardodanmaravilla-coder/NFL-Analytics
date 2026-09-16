"""Spread ATS v3: modela el MARGEN RESIDUAL contra la linea de mercado.
Validation only. Walk-forward estricto; 2026 excluido; sin veto MC.
Target historico: result - spread_line. Predice cuantos puntos supera/no supera el local la linea.
La probabilidad ATS sale de la distribucion OOS de residuos del modelo, no de Moneyline.
"""
import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from modules.nfl_calibration import historico_antes
from modules.nfl_ml_engine import PredictorNFL_ML

MIN_P=58.;MIN_EDGE=3.;MIN_EV=3.;MIN_RESID=80

def dec(x):
    x=float(x);return 1+x/100 if x>0 else 1+100/abs(x)
def nv(a,b):
    da,db=dec(a),dec(b);ia,ib=1/da,1/db;s=ia+ib;return ia/s,ib/s

def training(games,pbp):
    b=PredictorNFL_ML();f=b.construir_features_pregame(games,pbp)
    cols=b._base_feature_names();pc=b._pbp_feature_names();
    if not pbp.empty and all(c in f.columns for c in pc):cols+=pc
    e=games[['game_id','spread_line','result']].copy();f=f.merge(e,on='game_id',how='left')
    f['market_spread_home']=pd.to_numeric(f.spread_line,errors='coerce');f['ats_margin']=pd.to_numeric(f.result,errors='coerce')-f.market_spread_home
    return b,f,cols+['market_spread_home']

def row(b,pbp,w,h,a,g,line,cols):
    H=b.historial_actual
    if h not in H or a not in H or len(H[h]['pf'])<4 or len(H[a]['pf'])<4:return None
    temp=pd.to_numeric(g.get('temp'),errors='coerce');wind=pd.to_numeric(g.get('wind'),errors='coerce');hr=pd.to_numeric(g.get('home_rest'),errors='coerce');ar=pd.to_numeric(g.get('away_rest'),errors='coerce')
    r={'week':float(w),'home_altitude':float(b.altitud_estadios.get(h,0)),'temp':0 if pd.isna(temp) else float(temp),'wind':0 if pd.isna(wind) else float(wind),'is_dome':int(str(g.get('roof','')).lower() in {'dome','closed','indoor','indoors'}),'temp_missing':int(pd.isna(temp)),'wind_missing':int(pd.isna(wind)),'home_rest':0 if pd.isna(hr) else float(hr),'away_rest':0 if pd.isna(ar) else float(ar),'market_spread_home':float(line)}
    r.update(b._features_equipo(H,h,'home'));r.update(b._features_equipo(H,a,'away'))
    if any(c.startswith('home_off_epa_play_') for c in cols):
        from modules.nfl_pbp_engine import features_pbp_actuales
        q=features_pbp_actuales(pbp,h,a)
        if q is None:return None
        r.update(q)
    x=pd.DataFrame([r]);return x[cols] if all(c in x.columns for c in cols) and not x[cols].isna().any(axis=None) else None

def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025]);raw=raw[raw.result.notna()].copy()
    if 'game_type' in raw:raw=raw[raw.game_type.isin(['REG','POST','WC','DIV','CON','SB'])]
    p='data/historico_nfl_pbp_team_game.csv';pbp=pd.read_csv(p) if os.path.exists(p) else pd.DataFrame();rows=[]
    for s in [2023,2024,2025]:
      for w in sorted(pd.to_numeric(raw[raw.season==s].week,errors='coerce').dropna().astype(int).unique()):
        past=historico_antes(raw,s,w);pp=historico_antes(pbp,s,w) if not pbp.empty else pd.DataFrame();b,tr,cols=training(past,pp);tr=tr.dropna(subset=cols+['ats_margin'])
        if len(tr)<250:continue
        cut=max(200,int(len(tr)*.70));fit=tr.iloc[:cut];cal=tr.iloc[cut:]
        if len(cal)<MIN_RESID:continue
        m0=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=10,random_state=91,n_jobs=1).fit(fit[cols],fit.ats_margin)
        resid=cal.ats_margin.to_numpy()-m0.predict(cal[cols]);resid=resid[np.isfinite(resid)]
        if len(resid)<MIN_RESID:continue
        m=RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=10,random_state=91,n_jobs=1).fit(tr[cols],tr.ats_margin)
        for _,g in raw[(raw.season==s)&(pd.to_numeric(raw.week,errors='coerce')==w)].iterrows():
          h,a=g.home_team,g.away_team;line=pd.to_numeric(g.spread_line,errors='coerce')
          if pd.isna(line) or str(g.get('location','')).lower()=='neutral':continue
          ho=pd.to_numeric(g.get('home_spread_odds'),errors='coerce');ao=pd.to_numeric(g.get('away_spread_odds'),errors='coerce')
          if pd.isna(ho) or pd.isna(ao):continue
          x=row(b,pp,w,h,a,g,line,cols)
          if x is None:continue
          mu=float(m.predict(x)[0]);ph=100*float(np.mean(mu+resid>0));pa=100-ph;mh,ma=nv(ho,ao)
          for side,prob,odd,imp,ln in [('H',ph,float(ho),mh,-float(line)),('A',pa,float(ao),ma,float(line))]:
            edge=prob-100*imp;ev=(prob/100*dec(odd)-1)*100
            if prob<MIN_P or edge<MIN_EDGE or ev<MIN_EV:continue
            cover=float(g.result)-float(line);cover=cover if side=='H' else -cover
            if abs(cover)<1e-9:continue
            win=int(cover>0);role='FAVORITE' if ln<0 else ('UNDERDOG' if ln>0 else 'PICKEM')
            rows.append({'season':s,'week':w,'game_id':g.game_id,'side':side,'role':role,'line':ln,'probability':prob,'edge':edge,'ev':ev,'odds':odd,'pred_ats_margin':mu,'resid_n':len(resid),'win':win,'return':dec(odd)-1 if win else -1.})
    o=pd.DataFrame(rows);o.to_csv('backtest_nfl_spread_market_residual_results.csv',index=False)
    if o.empty:raise SystemExit('No picks')
    print('\n=== MARKET RESIDUAL ATS ===');print({'n':len(o),'winrate':round(100*o.win.mean(),2),'roi':round(100*o['return'].mean(),2),'avg_p':round(o.probability.mean(),2)})
    print('\nBY SEASON');print(o.groupby('season').agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_p=('probability','mean')).to_string())
    print('\nROLE x SEASON');print(o.groupby(['season','role']).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_p=('probability','mean')).to_string())
if __name__=='__main__':main()
