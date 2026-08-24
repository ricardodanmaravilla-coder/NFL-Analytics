import math

import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor

from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo


def dec(am):
    try:
        if pd.isna(am) or float(am) == 0:
            return None
        x = float(am)
        return 1 + (x / 100.0 if x > 0 else 100.0 / abs(x))
    except Exception:
        return None


def no_vig(a, b):
    da, db = dec(a), dec(b)
    if da is None or db is None:
        return None, None
    ia, ib = 1/da, 1/db
    s = ia + ib
    return ia/s, ib/s


def p_gt(mean, threshold, sigma):
    if sigma is None or sigma <= 0:
        return None
    z = (float(mean) - float(threshold)) / float(sigma)
    return 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100


def norm2(a, b):
    if a is None or b is None:
        return None, None
    s = float(a) + float(b)
    return (float(a)/s*100, float(b)/s*100) if s > 0 else (None, None)


def choose(probs, odds_self, odds_other, min_prob, max_dis=15, min_edge=3, min_ev=3):
    probs = [float(x) for x in probs if x is not None]
    if len(probs) < 2 or max(probs) - min(probs) > max_dis:
        return None
    p = float(np.mean(probs))
    mkt, _ = no_vig(odds_self, odds_other)
    d = dec(odds_self)
    if mkt is None or d is None:
        return None
    edge = (p/100 - mkt) * 100
    ev = ((p/100) * d - 1) * 100
    if p >= min_prob and edge >= min_edge and ev >= min_ev:
        return p, d, edge, ev
    return None


def settle_decimal(win, loss, push, decimal):
    if push:
        return 0.0
    if win:
        return decimal - 1.0
    if loss:
        return -1.0
    return 0.0


def main():
    raw = nfl.import_schedules([2021, 2022, 2023, 2024, 2025])
    raw = raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:
        raw = raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()

    builder = PredictorNFL_ML()
    feat = builder.construir_features_pregame(raw)
    market_cols = ['game_id','spread_line','total_line','home_moneyline','away_moneyline','home_spread_odds','away_spread_odds','over_odds','under_odds','home_score','away_score']
    df = feat.merge(raw[[c for c in market_cols if c in raw.columns]], on='game_id', how='left')

    features = [
        'week','home_altitude','temp','wind','is_dome','temp_missing','wind_missing','home_rest','away_rest',
        'home_off_5','home_def_5','home_margin_5','home_total_5','home_off_17','home_def_17','home_margin_17','home_total_17','home_score_sd_17',
        'away_off_5','away_def_5','away_margin_5','away_total_5','away_off_17','away_def_17','away_margin_17','away_total_17','away_score_sd_17'
    ]
    df = df.dropna(subset=features + ['puntos_totales','margen_local'])
    train = df[df['season'] <= 2024].copy()
    test = df[df['season'] == 2025].copy()
    assert len(train) > 700 and len(test) > 150

    cal_cut = int(len(train) * 0.8)
    tm = RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,random_state=42,n_jobs=1)
    mm = RandomForestRegressor(n_estimators=250,max_depth=9,min_samples_leaf=6,random_state=43,n_jobs=1)
    tm.fit(train[features].iloc[:cal_cut], train['puntos_totales'].iloc[:cal_cut])
    mm.fit(train[features].iloc[:cal_cut], train['margen_local'].iloc[:cal_cut])
    sigma_t = float(np.std(train['puntos_totales'].iloc[cal_cut:].to_numpy() - tm.predict(train[features].iloc[cal_cut:]), ddof=1))
    sigma_m = float(np.std(train['margen_local'].iloc[cal_cut:].to_numpy() - mm.predict(train[features].iloc[cal_cut:]), ddof=1))
    tm.fit(train[features], train['puntos_totales'])
    mm.fit(train[features], train['margen_local'])

    test = test.sort_values(['week','game_id']).copy()
    test['pred_total'] = tm.predict(test[features])
    test['pred_margin'] = mm.predict(test[features])

    returns_ml, returns_ou = [], []
    n_ml = n_ou = 0
    wins_ml = wins_ou = 0

    for week, wk in test.groupby('week', sort=True):
        past = raw[(raw['season'] < 2025) | ((raw['season'] == 2025) & (raw['week'] < week))].copy()
        elo = MotorELONFL(); elo.actualizar_ratings(past)

        for _, r in wk.iterrows():
            home, away = r['home_team'], r['away_team']
            emp = simular_nfl_montecarlo(home, away, past, r.get('total_line'), r.get('spread_line'))
            if not emp.get('Disponible'):
                continue

            elo_h = elo.calcular_probabilidad_elo(elo.ratings.get(home,1500), elo.ratings.get(away,1500))*100
            ml_h = p_gt(r['pred_margin'], 0.0, sigma_m)
            emp_h, emp_a = norm2(emp['Moneyline'].get('Gana Local'), emp['Moneyline'].get('Gana Visita'))

            hm, am = r.get('home_moneyline'), r.get('away_moneyline')
            if pd.notna(hm) and pd.notna(am):
                ch = choose([elo_h, ml_h, emp_h], hm, am, 54.0)
                ca = choose([100-elo_h, 100-ml_h, emp_a], am, hm, 54.0)
                choices = [('H', ch, hm), ('A', ca, am)]
                choices = [x for x in choices if x[1] is not None]
                if choices:
                    side, c, odd = max(choices, key=lambda x: x[1][2] + x[1][3])
                    d = c[1]; hs, aws = r['home_score'], r['away_score']
                    win = (side=='H' and hs>aws) or (side=='A' and aws>hs)
                    loss = (side=='H' and hs<aws) or (side=='A' and aws<hs)
                    push = hs==aws
                    returns_ml.append(settle_decimal(win, loss, push, d)); n_ml += 1; wins_ml += int(win)

            line = r.get('total_line'); oo, uo = r.get('over_odds'), r.get('under_odds')
            if pd.notna(line) and pd.notna(oo) and pd.notna(uo):
                ml_o = p_gt(r['pred_total'], line, sigma_t)
                emp_o, emp_u = norm2(emp['Over_Under'].get('Prob Over'), emp['Over_Under'].get('Prob Under'))
                co = choose([ml_o, emp_o], oo, uo, 53.5)
                cu = choose([100-ml_o, emp_u], uo, oo, 53.5)
                choices = [('O',co,oo),('U',cu,uo)]
                choices = [x for x in choices if x[1] is not None]
                if choices:
                    side,c,odd=max(choices,key=lambda x:x[1][2]+x[1][3]); d=c[1]
                    actual=r['home_score']+r['away_score']
                    win=(side=='O' and actual>line) or (side=='U' and actual<line)
                    loss=(side=='O' and actual<line) or (side=='U' and actual>line)
                    push=actual==line
                    returns_ou.append(settle_decimal(win,loss,push,d)); n_ou+=1; wins_ou+=int(win)

    roi_ml = (sum(returns_ml)/len(returns_ml)*100) if returns_ml else 0.0
    roi_ou = (sum(returns_ou)/len(returns_ou)*100) if returns_ou else 0.0
    print(f'sigma_total={sigma_t:.4f} sigma_margin={sigma_m:.4f}')
    print(f'2025_moneyline_picks={n_ml} wins={wins_ml} winrate={(wins_ml/n_ml if n_ml else 0):.4f} roi={roi_ml:.2f}%')
    print(f'2025_ou_picks={n_ou} wins={wins_ou} winrate={(wins_ou/n_ou if n_ou else 0):.4f} roi={roi_ou:.2f}%')

    # Guardrail: we require enough observations to judge; ROI itself is reported,
    # not hard-coded as a pass/fail until thresholds are reviewed.
    assert len(test) >= 150


if __name__ == '__main__':
    main()
