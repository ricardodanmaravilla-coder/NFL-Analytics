import math

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_pbp_max_engine import PredictorNFL_PBPMax
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo


def dec(am):
    try:
        x=float(am)
        if pd.isna(x) or x == 0: return None
        return 1 + (x/100 if x > 0 else 100/abs(x))
    except Exception: return None


def no_vig(a,b):
    da,db=dec(a),dec(b)
    if da is None or db is None: return None,None
    ia,ib=1/da,1/db; s=ia+ib
    return ia/s,ib/s


def p_margin(mean,sigma):
    if sigma is None or sigma <= 0: return None
    z=float(mean)/float(sigma)
    return 50*(1+math.erf(z/math.sqrt(2)))


def two_way(a,b):
    if a is None or b is None: return None,None
    s=float(a)+float(b)
    return (100*float(a)/s,100*float(b)/s) if s>0 else (None,None)


def choose(probs,odd_self,odd_other):
    probs=[float(p) for p in probs if p is not None]
    if len(probs)<3 or max(probs)-min(probs)>15: return None
    p=float(np.mean(probs)); m,_=no_vig(odd_self,odd_other); d=dec(odd_self)
    if m is None or d is None: return None
    edge=(p/100-m)*100; ev=((p/100)*d-1)*100
    if p>=54 and edge>=3 and ev>=3: return p,d,edge,ev
    return None


def settle(side,row,d):
    hs,aws=float(row.home_score),float(row.away_score)
    win=(side=='H' and hs>aws) or (side=='A' and aws>hs)
    return d-1 if win else -1


def prepare(raw,pbp):
    base=PredictorNFL_ML()
    maxb=PredictorNFL_PBPMax()
    fbase=base.construir_features_pregame(raw,pbp)
    fmax=maxb.construir_features_pregame(raw,pbp)
    markets=['game_id','home_score','away_score','home_moneyline','away_moneyline','total_line','spread_line']
    src=raw[[c for c in markets if c in raw.columns]].drop_duplicates('game_id')
    fbase=fbase.merge(src,on='game_id',how='left',suffixes=('','_mkt'))
    fmax=fmax.merge(src,on='game_id',how='left',suffixes=('','_mkt'))
    return base,maxb,fbase,fmax


def evaluate_season(raw,pbp,base,maxb,fbase,fmax,season):
    base_features=base._base_feature_names()+base._pbp_feature_names()
    max_features=base._base_feature_names()
    selected_raw=[]
    for w in (4,8):
        for side in ('home','away'):
            for metric in ('off_epa_play','off_success_rate','pass_epa','rush_epa','explosive_rate','def_epa_allowed','def_success_allowed','pressure_rate'):
                selected_raw.append(f'{side}_{metric}_{w}')
    max_features += selected_raw + maxb._matchup_names()

    train_b=fbase[fbase.season < season].dropna(subset=base_features+['margen_local']).copy()
    test_b=fbase[fbase.season == season].dropna(subset=base_features+['margen_local']).copy()
    train_x=fmax[fmax.season < season].dropna(subset=max_features+['margen_local']).copy()
    test_x=fmax[fmax.season == season].dropna(subset=max_features+['margen_local']).copy()
    common=set(test_b.game_id.astype(str)) & set(test_x.game_id.astype(str))
    test_b=test_b[test_b.game_id.astype(str).isin(common)].sort_values(['week','game_id']).copy()
    test_x=test_x[test_x.game_id.astype(str).isin(common)].set_index('game_id').loc[test_b.game_id].reset_index()
    assert len(test_b) >= 180

    cut=max(150,int(len(train_b)*0.8))
    reg=RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,random_state=43,n_jobs=1)
    reg.fit(train_b[base_features].iloc[:cut],train_b['margen_local'].iloc[:cut])
    sigma=float(np.std(train_b['margen_local'].iloc[cut:].to_numpy()-reg.predict(train_b[base_features].iloc[cut:]),ddof=1))
    reg.fit(train_b[base_features],train_b['margen_local'])
    pb=reg.predict(test_b[base_features])
    p_current=np.array([p_margin(x,sigma) for x in pb])

    clf=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=0.20,max_iter=3000,random_state=73))])
    yx=(train_x['margen_local']>0).astype(int)
    clf.fit(train_x[max_features],yx)
    p_max=clf.predict_proba(test_x[max_features])[:,1]*100

    y=(test_b['margen_local'].to_numpy()>0).astype(int)
    metrics={
        'season':season,
        'games':len(y),
        'current_acc':float(np.mean((p_current>=50)==y)),
        'current_brier':float(np.mean((p_current/100-y)**2)),
        'max_acc':float(np.mean((p_max>=50)==y)),
        'max_brier':float(np.mean((p_max/100-y)**2)),
    }

    returns_c=[]; returns_x=[]; wins_c=wins_x=0
    test_b=test_b.reset_index(drop=True)
    for i,r in test_b.iterrows():
        week=int(r.week); home=r.home_team; away=r.away_team
        past=raw[(raw.season<season)|((raw.season==season)&(raw.week<week))].copy()
        emp=simular_nfl_montecarlo(home,away,past,r.get('total_line'),r.get('spread_line'))
        if not emp.get('Disponible'): continue
        elo=MotorELONFL(); elo.actualizar_ratings(past)
        pelo=100*elo.calcular_probabilidad_elo(elo.ratings.get(home,1500),elo.ratings.get(away,1500))
        pe_h,pe_a=two_way(emp['Moneyline'].get('Gana Local'),emp['Moneyline'].get('Gana Visita'))
        hm,am=r.get('home_moneyline'),r.get('away_moneyline')
        if pd.isna(hm) or pd.isna(am): continue
        for pml,arr,key in [(p_current[i],returns_c,'c'),(p_max[i],returns_x,'x')]:
            ch=choose([pelo,pml,pe_h],hm,am); ca=choose([100-pelo,100-pml,pe_a],am,hm)
            choices=[('H',ch),('A',ca)]; choices=[z for z in choices if z[1] is not None]
            if not choices: continue
            side,c=max(choices,key=lambda z:z[1][2]+z[1][3]); ret=settle(side,r,c[1]); arr.append(ret)
            if ret>0:
                if key=='c': wins_c+=1
                else: wins_x+=1

    metrics.update({
        'current_picks':len(returns_c),'current_wins':wins_c,'current_roi':100*sum(returns_c)/len(returns_c) if returns_c else 0,
        'max_picks':len(returns_x),'max_wins':wins_x,'max_roi':100*sum(returns_x)/len(returns_x) if returns_x else 0,
    })
    print(metrics)
    return metrics


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw: raw=raw[raw.game_type.isin(['REG','POST','WC','DIV','CON','SB'])]
    pbp=pd.read_csv('data/historico_nfl_pbp_team_game.csv')
    base,maxb,fbase,fmax=prepare(raw,pbp)
    out=[evaluate_season(raw,pbp,base,maxb,fbase,fmax,s) for s in (2023,2024,2025)]
    total_games=sum(x['games'] for x in out)
    cur_acc=sum(x['current_acc']*x['games'] for x in out)/total_games
    max_acc=sum(x['max_acc']*x['games'] for x in out)/total_games
    cur_brier=sum(x['current_brier']*x['games'] for x in out)/total_games
    max_brier=sum(x['max_brier']*x['games'] for x in out)/total_games
    cur_returns=sum(x['current_picks'] for x in out); max_returns=sum(x['max_picks'] for x in out)
    print({'walkforward_games':total_games,'current_acc':cur_acc,'max_acc':max_acc,'current_brier':cur_brier,'max_brier':max_brier,'current_picks':cur_returns,'max_picks':max_returns})
    print(f'delta_accuracy_pp={(max_acc-cur_acc)*100:.2f}')
    print(f'delta_brier={max_brier-cur_brier:.5f}')
    assert total_games >= 600


if __name__=='__main__': main()
