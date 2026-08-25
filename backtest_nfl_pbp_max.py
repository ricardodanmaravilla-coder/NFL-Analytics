import math

import numpy as np
import pandas as pd
import nfl_data_py as nfl

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


def evaluate(engine,raw,pbp,label,use_classifier=False):
    assert engine.entrenar(raw,df_pbp_team_game=pbp)
    test=raw[(raw.season==2025)&raw.home_score.notna()&raw.away_score.notna()].copy()
    test=test.sort_values(['week','game_id'])
    returns=[]; wins=0; picks=0; winner=[]; brier=[]

    for week,wk in test.groupby('week',sort=True):
        past=raw[(raw.season<2025)|((raw.season==2025)&(raw.week<week))].copy()
        elo=MotorELONFL(); elo.actualizar_ratings(past)
        for _,r in wk.iterrows():
            home,away=r.home_team,r.away_team
            pred=engine.predecir_contexto(week,home,away,r.get('temp'),r.get('wind'),str(r.get('roof','')).lower() in {'dome','closed'},r.get('home_rest'),r.get('away_rest'))
            emp=simular_nfl_montecarlo(home,away,past,r.get('total_line'),r.get('spread_line'))
            if not pred or not emp.get('Disponible'): continue
            if use_classifier and pred.get('Probabilidad_Local_ML') is not None:
                pml=float(pred['Probabilidad_Local_ML'])
            else:
                pml=p_margin(pred.get('ML_Margen_Local_Esperado'),pred.get('Sigma_Margen_OOS'))
            if pml is None: continue
            y=1 if r.home_score>r.away_score else 0
            winner.append(int((pml>=50)==bool(y))); brier.append((pml/100-y)**2)
            pe_h,pe_a=two_way(emp['Moneyline'].get('Gana Local'),emp['Moneyline'].get('Gana Visita'))
            pelo=100*elo.calcular_probabilidad_elo(elo.ratings.get(home,1500),elo.ratings.get(away,1500))
            hm,am=r.get('home_moneyline'),r.get('away_moneyline')
            if pd.isna(hm) or pd.isna(am): continue
            ch=choose([pelo,pml,pe_h],hm,am); ca=choose([100-pelo,100-pml,pe_a],am,hm)
            choices=[('H',ch),('A',ca)]; choices=[x for x in choices if x[1] is not None]
            if choices:
                side,c=max(choices,key=lambda x:x[1][2]+x[1][3]); ret=settle(side,r,c[1])
                returns.append(ret); picks+=1; wins+=int(ret>0)

    roi=100*sum(returns)/len(returns) if returns else 0
    out={'label':label,'winner_acc':float(np.mean(winner)) if winner else None,'brier':float(np.mean(brier)) if brier else None,'picks':picks,'wins':wins,'winrate':wins/picks if picks else 0,'roi':roi}
    print(out)
    return out


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw: raw=raw[raw.game_type.isin(['REG','POST','WC','DIV','CON','SB'])]
    pbp=pd.read_csv('data/historico_nfl_pbp_team_game.csv')
    p1=evaluate(PredictorNFL_ML(),raw,pbp,'PBP-current',False)
    p2=evaluate(PredictorNFL_PBPMax(),raw,pbp,'PBP-Max',True)
    print(f"delta_winner_pp={(p2['winner_acc']-p1['winner_acc'])*100:.2f}")
    print(f"delta_brier={p2['brier']-p1['brier']:.4f}")
    print(f"delta_roi_pp={p2['roi']-p1['roi']:.2f}")
    assert p2['winner_acc'] is not None and p2['brier'] is not None


if __name__=='__main__': main()
