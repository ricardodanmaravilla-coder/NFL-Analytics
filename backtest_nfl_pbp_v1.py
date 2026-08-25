import math

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo


def dec(am):
    try:
        x=float(am)
        if pd.isna(x) or x == 0: return None
        return 1+x/100 if x>0 else 1+100/abs(x)
    except Exception: return None


def no_vig(a,b):
    da,db=dec(a),dec(b)
    if da is None or db is None: return None,None
    ia,ib=1/da,1/db; s=ia+ib
    return ia/s,ib/s


def p_gt(mean,threshold,sigma):
    if sigma is None or sigma<=0: return None
    z=(float(mean)-float(threshold))/float(sigma)
    return 50*(1+math.erf(z/math.sqrt(2)))


def norm2(a,b):
    if a is None or b is None: return None,None
    s=float(a)+float(b)
    return (100*float(a)/s,100*float(b)/s) if s>0 else (None,None)


def choose(probs,odds_self,odds_other,min_prob,max_dis=15,min_edge=3,min_ev=3):
    probs=[float(x) for x in probs if x is not None]
    if len(probs)<2 or max(probs)-min(probs)>max_dis: return None
    p=float(np.mean(probs)); mkt,_=no_vig(odds_self,odds_other); d=dec(odds_self)
    if mkt is None or d is None: return None
    edge=(p/100-mkt)*100; ev=((p/100)*d-1)*100
    return (p,d,edge,ev) if p>=min_prob and edge>=min_edge and ev>=min_ev else None


def settle(win,push,d):
    if push: return 0.0
    return d-1 if win else -1.0


BASE_FEATURES=[
    'week','home_altitude','temp','wind','is_dome','temp_missing','wind_missing','home_rest','away_rest',
    'home_off_5','home_def_5','home_margin_5','home_total_5','home_off_17','home_def_17','home_margin_17','home_total_17','home_score_sd_17',
    'away_off_5','away_def_5','away_margin_5','away_total_5','away_off_17','away_def_17','away_margin_17','away_total_17','away_score_sd_17'
]


def pbp_subset(mode):
    if mode == 'BASE': return []
    if mode == 'EPA8':
        metrics=['off_epa_play','off_success_rate','pass_epa','def_epa_allowed','def_success_allowed','pressure_rate']
        windows=[8]
    elif mode == 'EPA48':
        metrics=['off_epa_play','off_success_rate','pass_epa','def_epa_allowed','def_success_allowed','pressure_rate']
        windows=[4,8]
    elif mode == 'CORE8':
        metrics=['off_epa_play','off_success_rate','pass_epa','rush_epa','explosive_rate','sack_rate_allowed','def_epa_allowed','def_success_allowed','def_explosive_allowed','pressure_rate']
        windows=[8]
    else:
        metrics=['off_epa_play','off_success_rate','pass_epa','rush_epa','explosive_rate','sack_rate_allowed','plays','def_epa_allowed','def_success_allowed','def_explosive_allowed','pressure_rate']
        windows=[4,8]
    return [f'{side}_{m}_{w}' for side in ['home','away'] for m in metrics for w in windows]


def evaluate(raw,pbp,label):
    builder=PredictorNFL_ML()
    feat=builder.construir_features_pregame(raw,pbp if label!='BASE' else None)
    market_cols=['game_id','spread_line','total_line','home_moneyline','away_moneyline','over_odds','under_odds','home_score','away_score']
    df=feat.merge(raw[[c for c in market_cols if c in raw.columns]],on='game_id',how='left')
    features=BASE_FEATURES + [c for c in pbp_subset(label) if c in df.columns]
    df=df.dropna(subset=features+['puntos_totales','margen_local'])
    train=df[df['season']<=2024].copy(); test=df[df['season']==2025].copy()
    assert len(train)>600 and len(test)>120

    cal=int(len(train)*0.8)
    tm=RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,random_state=42,n_jobs=1)
    mm=RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,random_state=43,n_jobs=1)
    tm.fit(train[features].iloc[:cal],train['puntos_totales'].iloc[:cal])
    mm.fit(train[features].iloc[:cal],train['margen_local'].iloc[:cal])
    sig_t=float(np.std(train['puntos_totales'].iloc[cal:].to_numpy()-tm.predict(train[features].iloc[cal:]),ddof=1))
    sig_m=float(np.std(train['margen_local'].iloc[cal:].to_numpy()-mm.predict(train[features].iloc[cal:]),ddof=1))
    tm.fit(train[features],train['puntos_totales']); mm.fit(train[features],train['margen_local'])
    test=test.sort_values(['week','game_id']).copy()
    test['pred_total']=tm.predict(test[features]); test['pred_margin']=mm.predict(test[features])

    total_mae=mean_absolute_error(test['puntos_totales'],test['pred_total'])
    margin_mae=mean_absolute_error(test['margen_local'],test['pred_margin'])
    wm=test['margen_local']!=0
    winner=float(np.mean((test.loc[wm,'pred_margin']>0)==(test.loc[wm,'margen_local']>0)))
    sm=test['spread_line'].notna() & ((test['margen_local']-test['spread_line'])!=0)
    ats=float(np.mean((test.loc[sm,'pred_margin']>test.loc[sm,'spread_line'])==((test.loc[sm,'margen_local']-test.loc[sm,'spread_line'])>0))) if sm.any() else np.nan
    om=test['total_line'].notna() & ((test['puntos_totales']-test['total_line'])!=0)
    ou_dir=float(np.mean((test.loc[om,'pred_total']>test.loc[om,'total_line'])==((test.loc[om,'puntos_totales']-test.loc[om,'total_line'])>0))) if om.any() else np.nan

    ret_ml=[]; ret_ou=[]; wins_ml=wins_ou=0
    for week,wk in test.groupby('week',sort=True):
        past=raw[(raw['season']<2025)|((raw['season']==2025)&(raw['week']<week))].copy()
        elo=MotorELONFL(); elo.actualizar_ratings(past)
        for _,r in wk.iterrows():
            home,away=r['home_team'],r['away_team']
            emp=simular_nfl_montecarlo(home,away,past,r.get('total_line'),r.get('spread_line'))
            if not emp.get('Disponible'): continue
            elo_h=100*elo.calcular_probabilidad_elo(elo.ratings.get(home,1500),elo.ratings.get(away,1500))
            ml_h=p_gt(r['pred_margin'],0,sig_m); emp_h,emp_a=norm2(emp['Moneyline'].get('Gana Local'),emp['Moneyline'].get('Gana Visita'))
            hm,am=r.get('home_moneyline'),r.get('away_moneyline')
            if pd.notna(hm) and pd.notna(am):
                ch=choose([elo_h,ml_h,emp_h],hm,am,54); ca=choose([100-elo_h,100-ml_h,emp_a],am,hm,54)
                choices=[x for x in [('H',ch,hm),('A',ca,am)] if x[1] is not None]
                if choices:
                    side,c,_=max(choices,key=lambda x:x[1][2]+x[1][3]); win=(side=='H' and r['home_score']>r['away_score']) or (side=='A' and r['away_score']>r['home_score']); push=r['home_score']==r['away_score']
                    ret_ml.append(settle(win,push,c[1])); wins_ml+=int(win)
            line=r.get('total_line'); oo,uo=r.get('over_odds'),r.get('under_odds')
            if pd.notna(line) and pd.notna(oo) and pd.notna(uo):
                mo=p_gt(r['pred_total'],line,sig_t); eo,eu=norm2(emp['Over_Under'].get('Prob Over'),emp['Over_Under'].get('Prob Under'))
                co=choose([mo,eo],oo,uo,53.5); cu=choose([100-mo,eu],uo,oo,53.5)
                choices=[x for x in [('O',co,oo),('U',cu,uo)] if x[1] is not None]
                if choices:
                    side,c,_=max(choices,key=lambda x:x[1][2]+x[1][3]); actual=r['home_score']+r['away_score']; win=(side=='O' and actual>line) or (side=='U' and actual<line); push=actual==line
                    ret_ou.append(settle(win,push,c[1])); wins_ou+=int(win)

    roi_ml=100*sum(ret_ml)/len(ret_ml) if ret_ml else 0.0
    roi_ou=100*sum(ret_ou)/len(ret_ou) if ret_ou else 0.0
    result={'label':label,'games':len(df),'test':len(test),'features':len(features),'total_mae':total_mae,'margin_mae':margin_mae,'winner_acc':winner,'ats_acc':ats,'ou_dir_acc':ou_dir,'ml_picks':len(ret_ml),'ml_wins':wins_ml,'ml_roi':roi_ml,'ou_picks':len(ret_ou),'ou_wins':wins_ou,'ou_roi':roi_ou,'sigma_total':sig_t,'sigma_margin':sig_m}
    print(result)
    return result


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns: raw=raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    pbp=pd.read_csv('data/historico_nfl_pbp_team_game.csv')
    results={}
    for label in ['BASE','EPA8','EPA48','CORE8','FULL']:
        results[label]=evaluate(raw,pbp,label)
    base=results['BASE']
    for label in ['EPA8','EPA48','CORE8','FULL']:
        r=results[label]
        print(f"{label}_delta_total_mae={r['total_mae']-base['total_mae']:.4f} delta_margin_mae={r['margin_mae']-base['margin_mae']:.4f} delta_winner_pp={(r['winner_acc']-base['winner_acc'])*100:.2f} delta_ats_pp={(r['ats_acc']-base['ats_acc'])*100:.2f} delta_ou_pp={(r['ou_dir_acc']-base['ou_dir_acc'])*100:.2f} delta_ml_roi_pp={r['ml_roi']-base['ml_roi']:.2f} delta_ou_roi_pp={r['ou_roi']-base['ou_roi']:.2f}")
    assert min(r['test'] for r in results.values()) >= 150


if __name__=='__main__':
    main()
