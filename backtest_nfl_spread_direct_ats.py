"""Walk-forward experimental para Spread NFL modelado DIRECTAMENTE contra la linea.

No cambia produccion. En vez de inferir ATS desde el modelo de margen/Moneyline y
usar Monte Carlo como veto, entrena un clasificador cuyo target es cubrir la linea
publicada. La linea nflverse es positiva cuando el local es favorito. Cada semana
se entrena solo con juegos anteriores. 2026 queda excluido.
"""
import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestClassifier

from modules.nfl_calibration import historico_antes
from modules.nfl_ml_engine import PredictorNFL_ML

MIN_P=58.0
MIN_EDGE=3.0
MIN_EV=3.0


def dec(am):
    try:
        x=float(am)
        if pd.isna(x) or x==0:return None
        return 1+x/100 if x>0 else 1+100/abs(x)
    except Exception:return None


def no_vig(a,b):
    da,db=dec(a),dec(b)
    if da is None or db is None:return None,None
    ia,ib=1/da,1/db;s=ia+ib
    return (ia/s,ib/s) if s else (None,None)


def make_training(games,pbp):
    base=PredictorNFL_ML()
    feats=base.construir_features_pregame(games,pbp)
    if feats.empty:return base,feats,[]
    cols=base._base_feature_names()
    pcols=base._pbp_feature_names()
    use_pbp=bool(pbp is not None and not pbp.empty and all(c in feats.columns for c in pcols))
    if use_pbp: cols=cols+pcols
    extra=games[[c for c in ['game_id','spread_line','result','home_score','away_score'] if c in games.columns]].copy()
    if 'result' not in extra.columns and {'home_score','away_score'}.issubset(extra.columns):
        extra['result']=pd.to_numeric(extra.home_score,errors='coerce')-pd.to_numeric(extra.away_score,errors='coerce')
    feats=feats.merge(extra[['game_id','spread_line','result']],on='game_id',how='left')
    feats['market_spread_home']=pd.to_numeric(feats.spread_line,errors='coerce')
    feats['cover_margin']=pd.to_numeric(feats.result,errors='coerce')-feats.market_spread_home
    feats=feats[feats.cover_margin.abs()>1e-9].copy() # pushes fuera del target binario
    feats['home_cover']=(feats.cover_margin>0).astype(int)
    return base,feats,cols+['market_spread_home']


def context_row(base,pbp,week,home,away,g,threshold,cols):
    hist=base.historial_actual
    if home not in hist or away not in hist or len(hist[home]['pf'])<4 or len(hist[away]['pf'])<4:return None
    temp=pd.to_numeric(g.get('temp'),errors='coerce');wind=pd.to_numeric(g.get('wind'),errors='coerce')
    hr=pd.to_numeric(g.get('home_rest'),errors='coerce');ar=pd.to_numeric(g.get('away_rest'),errors='coerce')
    row={'week':float(week),'home_altitude':float(base.altitud_estadios.get(home,0.0)),
         'temp':0.0 if pd.isna(temp) else float(temp),'wind':0.0 if pd.isna(wind) else float(wind),
         'is_dome':int(str(g.get('roof','')).lower() in {'dome','closed','indoor','indoors'}),
         'temp_missing':int(pd.isna(temp)),'wind_missing':int(pd.isna(wind)),
         'home_rest':0.0 if pd.isna(hr) else float(hr),'away_rest':0.0 if pd.isna(ar) else float(ar),
         'market_spread_home':float(threshold)}
    row.update(base._features_equipo(hist,home,'home'));row.update(base._features_equipo(hist,away,'away'))
    if any(c.startswith('home_off_epa_play_') for c in cols):
        from modules.nfl_pbp_engine import features_pbp_actuales
        p=features_pbp_actuales(pbp,home,away)
        if p is None:return None
        row.update(p)
    x=pd.DataFrame([row])
    if any(c not in x.columns for c in cols):return None
    x=x[cols]
    return None if x.isna().any(axis=None) else x


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:raw=raw[raw.game_type.isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    pbp_path='data/historico_nfl_pbp_team_game.csv';pbp=pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    rows=[]
    for season in [2023,2024,2025]:
        sg=raw[raw.season==season]
        for week in sorted(pd.to_numeric(sg.week,errors='coerce').dropna().astype(int).unique()):
            past=historico_antes(raw,season,week);pp=historico_antes(pbp,season,week) if not pbp.empty else pd.DataFrame()
            base,tr,cols=make_training(past,pp)
            if len(tr)<250 or not cols:continue
            tr=tr.dropna(subset=cols+['home_cover'])
            if len(tr)<250:continue
            model=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=10,class_weight='balanced',random_state=71,n_jobs=1)
            model.fit(tr[cols],tr.home_cover)
            wk=sg[pd.to_numeric(sg.week,errors='coerce')==week]
            for _,g in wk.iterrows():
                home,away=g.get('home_team'),g.get('away_team')
                threshold=pd.to_numeric(g.get('spread_line'),errors='coerce')
                if not home or not away or pd.isna(threshold) or str(g.get('location','')).lower()=='neutral':continue
                sho=pd.to_numeric(g.get('home_spread_odds'),errors='coerce');sao=pd.to_numeric(g.get('away_spread_odds'),errors='coerce')
                if pd.isna(sho) or pd.isna(sao):continue
                x=context_row(base,pp,week,home,away,g,float(threshold),cols)
                if x is None:continue
                ph=100*float(model.predict_proba(x)[0,1]);pa=100-ph
                result=float(g.result)
                for side,p,odd,other,line in [('H',ph,float(sho),float(sao),-float(threshold)),('A',pa,float(sao),float(sho),float(threshold))]:
                    m,_=no_vig(odd,other);d=dec(odd)
                    if m is None or d is None:continue
                    edge=(p/100-m)*100;ev=(p/100*d-1)*100
                    if p<MIN_P or edge<MIN_EDGE or ev<MIN_EV:continue
                    cover=(result-float(threshold)) if side=='H' else (float(threshold)-result)
                    if abs(cover)<1e-9:continue
                    win=int(cover>0);role='FAVORITE' if line<0 else ('UNDERDOG' if line>0 else 'PICKEM')
                    rows.append({'season':season,'week':week,'game_id':g.get('game_id'),'side':side,'role':role,'line':line,
                                 'probability':p,'edge':edge,'ev':ev,'odds':odd,'win':win,'return':d-1 if win else -1.0})
    out=pd.DataFrame(rows);out.to_csv('backtest_nfl_spread_direct_ats_results.csv',index=False)
    if out.empty:raise SystemExit('No direct ATS picks generated')
    print('\n=== DIRECT ATS SPREAD WALK-FORWARD ===')
    print({'n':len(out),'winrate':round(100*out.win.mean(),2),'roi':round(100*out['return'].mean(),2),'avg_p':round(out.probability.mean(),2)})
    print('\nBY ROLE')
    for role,g in out.groupby('role'):print(role,{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2),'avg_p':round(g.probability.mean(),2)})
    print('\nBY SEASON')
    for s,g in out.groupby('season'):print(int(s),{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2)})
    print('\nROLE x SEASON')
    print(out.groupby(['season','role']).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean')).to_string())

if __name__=='__main__':main()
