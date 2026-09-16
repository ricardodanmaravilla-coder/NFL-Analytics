"""Audita TODAS las oportunidades ATS v1 antes/despues de calibracion causal.
No modifica produccion. 2026 excluido. Explica por que 2024-25 quedan suprimidos.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import backtest_nfl_spread_direct_ats as base

MIN_HISTORY=30


def dec(am):
    x=float(am); return 1+x/100 if x>0 else 1+100/abs(x)

def calibrate(past,raw_p):
    if len(past)<MIN_HISTORY or past.win.nunique()<2:return raw_p
    pp=np.clip(past.probability.to_numpy()/100,1e-5,1-1e-5)
    x=np.log(pp/(1-pp))
    m=LogisticRegression(C=.25,solver='lbfgs',max_iter=1000).fit(x.reshape(-1,1),past.win.astype(int))
    p=np.clip(raw_p/100,1e-5,1-1e-5);z=np.log(p/(1-p))
    return 100*float(m.predict_proba([[z]])[0,1])

def main():
    base.main()
    df=pd.read_csv('backtest_nfl_spread_direct_ats_results.csv').sort_values(['season','week','game_id','side']).reset_index(drop=True)
    hist=[];rows=[]
    for _,r in df.iterrows():
        past=pd.DataFrame(hist); cp=calibrate(past,float(r.probability))
        market=(float(r.probability)-float(r.edge))/100;d=dec(r.odds)
        ce=(cp/100-market)*100; cev=(cp/100*d-1)*100
        q=r.to_dict();q.update(cal_probability=cp,cal_edge=ce,cal_ev=cev,cal_history_n=len(past),pass_p=int(cp>=58),pass_edge=int(ce>=3),pass_ev=int(cev>=3),accepted_cal=int(cp>=58 and ce>=3 and cev>=3))
        rows.append(q);hist.append(r.to_dict())
    a=pd.DataFrame(rows);a.to_csv('diagnose_nfl_ats_calibration_all.csv',index=False)
    def summ(g):
        return pd.Series({'n':len(g),'raw_p':g.probability.mean(),'cal_p':g.cal_probability.mean(),'cal_p_min':g.cal_probability.min(),'cal_p_max':g.cal_probability.max(),'actual_wr':100*g.win.mean(),'pass_p':g.pass_p.sum(),'pass_edge':g.pass_edge.sum(),'pass_ev':g.pass_ev.sum(),'accepted':g.accepted_cal.sum()})
    s=a.groupby(['season','role']).apply(summ).reset_index();s.to_csv('diagnose_nfl_ats_calibration_summary.csv',index=False)
    print('\n=== CALIBRATION ALL CANDIDATES ===');print(s.to_string(index=False))
    print('\n=== BY SEASON ===')
    print(a.groupby('season').apply(summ).reset_index().to_string(index=False))
    print('\n=== CAL P QUANTILES ===')
    print(a.groupby('season').cal_probability.quantile([0,.1,.25,.5,.75,.9,1]).unstack().to_string())

if __name__=='__main__':main()
