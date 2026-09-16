"""ATS adaptive causal calibration experiment. Validation only; 2026 excluded.
Uses exponentially decayed prior OOS picks and shrinks calibrated probability toward raw.
No future outcomes are used for a decision.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import backtest_nfl_spread_direct_ats as base

MIN_HISTORY=30
HALF_LIFE=40.0
BLEND_RAW=0.50
MIN_P=58.0
MIN_EDGE=3.0
MIN_EV=3.0

def dec(am):
    x=float(am); return 1+x/100 if x>0 else 1+100/abs(x)

def adaptive(past, raw):
    if len(past)<MIN_HISTORY or past.win.nunique()<2:return raw
    pp=np.clip(past.probability.to_numpy()/100,1e-5,1-1e-5); x=np.log(pp/(1-pp))
    age=np.arange(len(past)-1,-1,-1,dtype=float); w=np.power(.5,age/HALF_LIFE)
    m=LogisticRegression(C=.25,solver='lbfgs',max_iter=1000).fit(x.reshape(-1,1),past.win.astype(int),sample_weight=w)
    p=np.clip(raw/100,1e-5,1-1e-5); z=np.log(p/(1-p)); cal=100*float(m.predict_proba([[z]])[0,1])
    return BLEND_RAW*raw+(1-BLEND_RAW)*cal

def main():
    base.main();df=pd.read_csv('backtest_nfl_spread_direct_ats_results.csv').sort_values(['season','week','game_id','side']).reset_index(drop=True)
    hist=[];allrows=[]
    for _,r in df.iterrows():
        past=pd.DataFrame(hist); cp=adaptive(past,float(r.probability)); market=(float(r.probability)-float(r.edge))/100; d=dec(r.odds)
        edge=(cp/100-market)*100;ev=(cp/100*d-1)*100;ok=cp>=MIN_P and edge>=MIN_EDGE and ev>=MIN_EV
        q=r.to_dict();q.update(adaptive_probability=cp,adaptive_edge=edge,adaptive_ev=ev,accepted_adaptive=int(ok),history_n=len(past));allrows.append(q);hist.append(r.to_dict())
    a=pd.DataFrame(allrows);a.to_csv('backtest_nfl_spread_ats_adaptive_all.csv',index=False);out=a[a.accepted_adaptive==1].copy();out.to_csv('backtest_nfl_spread_ats_adaptive_results.csv',index=False)
    print('\n=== ADAPTIVE ATS ===')
    print({'n':len(out),'winrate':round(100*out.win.mean(),2) if len(out) else None,'roi':round(100*out['return'].mean(),2) if len(out) else None})
    if len(out):
        print('\nBY SEASON');print(out.groupby('season').agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_p=('adaptive_probability','mean')).to_string())
        print('\nROLE x SEASON');print(out.groupby(['season','role']).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_p=('adaptive_probability','mean')).to_string())
    print('\nALL-CANDIDATE P DISTRIBUTION');print(a.groupby('season').adaptive_probability.agg(['count','min','median','mean','max']).to_string())

if __name__=='__main__':main()
