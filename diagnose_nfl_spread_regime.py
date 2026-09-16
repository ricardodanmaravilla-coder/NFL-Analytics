"""Diagnostico OOS del Direct ATS por temporada, rol, linea, probabilidad y mes.
No modifica produccion ni usa 2026. Busca explicar inestabilidad 2024 sin tunear sobre ella.
"""
import pandas as pd
import numpy as np
import backtest_nfl_spread_direct_ats as base


def stats(g):
    return pd.Series({'n':len(g),'winrate':g.win.mean(),'roi':g['return'].mean(),'avg_p':g.p.mean(),'avg_line':g.line.mean()})


def main():
    base.main()
    df=pd.read_csv('backtest_nfl_spread_direct_ats_results.csv')
    df=df.sort_values(['season','week','game_id','side']).copy()
    df['abs_line']=df.line.abs()
    df['line_bucket']=pd.cut(df.abs_line,[-.01,2.5,5.5,8.5,100],labels=['0-2.5','3-5.5','6-8.5','9+'])
    df['p_bucket']=pd.cut(df.p,[0,.60,.63,.66,.70,1],labels=['58-60','60-63','63-66','66-70','70+'],include_lowest=True)
    tables=[]
    for name,cols in [('season_role',['season','role']),('season_line',['season','line_bucket']),('season_p',['season','p_bucket'])]:
        t=df.groupby(cols,observed=True).apply(stats).reset_index(); t.insert(0,'view',name); tables.append(t)
    out=pd.concat(tables,ignore_index=True,sort=False)
    out.to_csv('diagnose_nfl_spread_regime_results.csv',index=False)
    print('\n=== REGIME DIAGNOSTICS ===')
    print(out.to_string(index=False))
    # Drift de las variables disponibles en el CSV de picks: compara distribuciones sin usar outcome para filtrar.
    drift=df.groupby('season').agg(n=('win','size'),avg_p=('p','mean'),sd_p=('p','std'),avg_abs_line=('abs_line','mean'),favorite_share=('role',lambda x:(x=='FAVORITE').mean()),avg_odds=('odds','mean'),avg_edge=('edge','mean'),avg_ev=('ev','mean')).reset_index()
    drift.to_csv('diagnose_nfl_spread_drift_results.csv',index=False)
    print('\n=== INPUT/PICK DRIFT ===')
    print(drift.to_string(index=False))

if __name__=='__main__': main()
