"""Calibracion temporal causal para picks ATS Direct v1.
Usa SOLO picks OOS anteriores para corregir probabilidad. 2026 excluido.
No modifica produccion. No selecciona umbrales mirando el periodo futuro.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import backtest_nfl_spread_direct_ats as base

MIN_HISTORY=30
MIN_P=58.0
MIN_EDGE=3.0
MIN_EV=3.0


def dec(am):
    x=float(am)
    return 1+x/100 if x>0 else 1+100/abs(x)


def implied_novig(row):
    # El CSV v1 ya contiene edge = raw_p - no-vig implied.
    return (float(row.probability)-float(row.edge))/100.0


def calibrate(past, raw_p):
    if len(past)<MIN_HISTORY or past.win.nunique()<2:
        return raw_p
    x=np.log(np.clip(past.probability.to_numpy()/100,1e-5,1-1e-5)/np.clip(1-past.probability.to_numpy()/100,1e-5,1))
    model=LogisticRegression(C=0.25,solver='lbfgs',max_iter=1000)
    model.fit(x.reshape(-1,1),past.win.astype(int))
    z=np.log(np.clip(raw_p/100,1e-5,1-1e-5)/np.clip(1-raw_p/100,1e-5,1))
    return 100*float(model.predict_proba([[z]])[0,1])


def main():
    base.main()
    df=pd.read_csv('backtest_nfl_spread_direct_ats_results.csv').sort_values(['season','week','game_id','side']).reset_index(drop=True)
    history=[]; rows=[]
    for _,r in df.iterrows():
        past=pd.DataFrame(history)
        cp=calibrate(past,float(r.probability))
        market=implied_novig(r); d=dec(r.odds)
        edge=(cp/100-market)*100; ev=(cp/100*d-1)*100
        accepted=cp>=MIN_P and edge>=MIN_EDGE and ev>=MIN_EV
        q=r.to_dict();q.update({'raw_probability':float(r.probability),'cal_probability':cp,'cal_edge':edge,'cal_ev':ev,'accepted_cal':int(accepted),'cal_history_n':len(past)})
        if accepted: rows.append(q)
        history.append(r.to_dict())
    out=pd.DataFrame(rows)
    out.to_csv('backtest_nfl_spread_ats_calibrated_results.csv',index=False)
    print('\n=== CAUSAL ATS CALIBRATION ===')
    if out.empty:
        print('No calibrated picks'); return
    print({'n':len(out),'winrate':round(100*out.win.mean(),2),'roi':round(100*out['return'].mean(),2),'avg_cal_p':round(out.cal_probability.mean(),2)})
    print('\nBY ROLE')
    for role,g in out.groupby('role'): print(role,{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2),'avg_cal_p':round(g.cal_probability.mean(),2)})
    print('\nBY SEASON')
    for s,g in out.groupby('season'): print(int(s),{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2)})
    print('\nROLE x SEASON')
    print(out.groupby(['season','role']).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_cal_p=('cal_probability','mean')).to_string())

if __name__=='__main__': main()
