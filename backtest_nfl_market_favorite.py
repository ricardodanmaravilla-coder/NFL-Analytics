"""Compara el filtro legado odd<0 contra favorito real no-vig, sin usar 2026."""
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from backtest_nfl_walkforward import evaluate_season, load_pbp


def summarize(df):
    if df is None or df.empty:
        return {"picks":0,"wins":0,"winrate":np.nan,"roi":np.nan}
    return {"picks":len(df),"wins":int(df['win'].sum()),"winrate":float(df['win'].mean()),"roi":float(100*df['return'].mean())}


def filter_legacy(df):
    return df[df['decimal'] < 2.0].copy()


def filter_true_market_favorite(df):
    x=df.copy()
    x['market_p_pct']=x['p']-x['edge']
    return x[x['market_p_pct'] > 50.0].copy()


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:
        raw=raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    pbp=load_pbp()
    all_rows=[]
    for season in [2023,2024,2025]:
        _,_,bets=evaluate_season(raw,pbp,season)
        legacy=filter_legacy(bets)
        truefav=filter_true_market_favorite(bets)
        a=summarize(legacy); b=summarize(truefav)
        print('SEASON',season,'LEGACY_NEGATIVE_ODDS',a,'TRUE_MARKET_FAVORITE',b)
        all_rows.append((season,a,b))
    legacy_all=pd.concat([filter_legacy(evaluate_season(raw,pbp,s)[2]) for s in [2023,2024,2025]],ignore_index=True)
    true_all=pd.concat([filter_true_market_favorite(evaluate_season(raw,pbp,s)[2]) for s in [2023,2024,2025]],ignore_index=True)
    print('TOTAL LEGACY',summarize(legacy_all))
    print('TOTAL TRUE_MARKET_FAVORITE',summarize(true_all))
    # Selección sólo 2023/2024; 2025 sigue siendo test de supervivencia.
    dev_true=all_rows[0][2]; val_true=all_rows[1][2]; test_true=all_rows[2][2]
    candidate=(dev_true['picks']>=10 and val_true['picks']>=8 and dev_true['roi']>0 and val_true['roi']>0 and min(dev_true['winrate'],val_true['winrate'])>=0.55)
    survives=bool(candidate and test_true['picks']>=6 and test_true['roi']>-5 and test_true['winrate']>=0.50)
    print('TRUE_FAVORITE_CANDIDATE',candidate,'SURVIVES_2025',survives)
    assert 2026 not in set(pd.to_numeric(raw['season'],errors='coerce').dropna().astype(int).unique())

if __name__=='__main__':
    main()
