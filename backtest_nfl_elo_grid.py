"""Grid temporal de Elo NFL. 2023 desarrollo, 2024 validación, 2025 test intocable."""
import math
import numpy as np
import pandas as pd
import nfl_data_py as nfl

K_GRID=[10,15,20,25,30]
HA_GRID=[30,40,48,55,65]
REG_GRID=[0.20,0.33,0.50]


def logloss(y,p):
    p=min(max(float(p),1e-6),1-1e-6)
    return -(y*math.log(p)+(1-y)*math.log(1-p))


def run_elo(df,k,home_adv,regress):
    ratings={t:1500.0 for t in sorted(set(df['home_team'].dropna())|set(df['away_team'].dropna()))}
    current=None; rows=[]
    x=df.sort_values([c for c in ['season','week','gameday','game_id'] if c in df.columns]).copy()
    for _,r in x.iterrows():
        season=int(r['season'])
        if current is not None and season>current:
            for t in ratings:
                ratings[t]=ratings[t]*(1.0-regress)+1500.0*regress
        current=season
        h,a=r['home_team'],r['away_team']
        eh,ea=ratings[h],ratings[a]
        p=1.0/(1.0+10.0**(-(((eh+home_adv)-ea)/400.0)))
        hs,aws=float(r['home_score']),float(r['away_score'])
        y=1.0 if hs>aws else (0.0 if aws>hs else 0.5)
        if season in [2023,2024,2025] and y!=0.5:
            rows.append({'season':season,'y':y,'p':p,'brier':(p-y)**2,'logloss':logloss(y,p),'correct':int((p>=0.5)==(y==1.0))})
        mov=abs(hs-aws)
        mult=np.log(max(mov,1.0)+1.0)*(2.2/(abs(eh-ea)*0.001+2.2))
        change=float(k)*mult*(y-p)
        ratings[h]+=change; ratings[a]-=change
    return pd.DataFrame(rows)


def summary(d,season):
    s=d[d['season']==season]
    return {'n':len(s),'brier':float(s['brier'].mean()),'logloss':float(s['logloss'].mean()),'accuracy':float(s['correct'].mean())}


def score(dev,val):
    # Menor es mejor; prioriza peor temporada y penaliza logloss.
    return max(dev['brier'],val['brier']) + 0.15*max(dev['logloss'],val['logloss'])


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:
        raw=raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    cand=[]
    for k in K_GRID:
        for ha in HA_GRID:
            for reg in REG_GRID:
                d=run_elo(raw,k,ha,reg)
                dev,val=summary(d,2023),summary(d,2024)
                sc=score(dev,val)
                cand.append({'k':k,'home_adv':ha,'regress':reg,'score':sc,'dev':dev,'val':val})
                print('ELO_GRID',k,ha,reg,'DEV',dev,'VAL',val,'score',sc)
    cand.sort(key=lambda z:z['score'])
    best=cand[0]
    bd=run_elo(raw,best['k'],best['home_adv'],best['regress'])
    test=summary(bd,2025)
    base=run_elo(raw,20,48,0.33)
    base_dev,base_val,base_test=summary(base,2023),summary(base,2024),summary(base,2025)
    print('SELECTED_WITHOUT_2025',best)
    print('SELECTED_2025_UNTOUCHED',test)
    print('BASE_20_48_033',base_dev,base_val,base_test)
    robust=(best['dev']['brier']<=base_dev['brier'] and best['val']['brier']<=base_val['brier'] and best['dev']['logloss']<=base_dev['logloss'] and best['val']['logloss']<=base_val['logloss'])
    print('ROBUST_CANDIDATE',robust)
    assert 2026 not in set(pd.to_numeric(raw['season'],errors='coerce').dropna().astype(int).unique())

if __name__=='__main__':
    main()
