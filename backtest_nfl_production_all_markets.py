"""Walk-forward de los TRES mercados con la misma semántica del scanner Cloud Run.

Objetivo: impedir que ML validado se use como prueba indirecta de Spread/Total.
Cada semana entrena solo con información anterior y reproduce probabilidades,
consenso, no-vig y filtros de producción. 2026 queda fuera como prueba prospectiva.
"""
import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from modules.nfl_calibration import empirical_residual_two_way, historico_antes
from modules.nfl_elo_engine import MotorELONFL
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo

MIN_P = 58.0
MIN_MC = 58.0
MIN_EDGE = 3.0
MIN_EV = 3.0
MAX_DISAGREE = 15.0


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
    return (ia/s,ib/s) if s else (None,None)


def norm2(a,b):
    if a is None or b is None: return None,None
    s=float(a)+float(b)
    return (100*float(a)/s,100*float(b)/s) if s else (None,None)


def accepted(p, mc, odd, other):
    if None in (p,mc): return None
    p=float(p); mc=float(mc)
    if (p>=50)!=(mc>=50) or abs(p-mc)>MAX_DISAGREE: return None
    m,_=no_vig(odd,other); d=dec(odd)
    if m is None or d is None: return None
    edge=(p/100-m)*100; ev=(p/100*d-1)*100
    if p<MIN_P or mc<MIN_MC or edge<MIN_EDGE or ev<MIN_EV: return None
    return p,d,edge,ev


def grade(market, side, hs, aws, line):
    margin=hs-aws; total=hs+aws
    if market=='ML':
        if hs==aws: return None
        return int((side=='H' and hs>aws) or (side=='A' and aws>hs))
    if market=='SPREAD':
        # line es la línea de apuesta de la casa: home -3 / away +3.
        adj=(margin+line) if side=='H' else (-margin+line)
    else:
        adj=(total-line) if side=='O' else (line-total)
    if abs(adj)<1e-9: return None
    return int(adj>0)


def add(rows, season, week, gid, market, side, p, mc, odd, other, hs, aws, line=0):
    x=accepted(p,mc,odd,other)
    if x is None: return
    prob,d,edge,ev=x; win=grade(market,side,hs,aws,line)
    if win is None: return
    rows.append({'season':season,'week':week,'game_id':gid,'market':market,'side':side,
                 'probability':prob,'mc_probability':mc,'edge':edge,'ev':ev,'odds':odd,
                 'win':win,'return':d-1 if win else -1.0})


def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:
        raw=raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    pbp_path='data/historico_nfl_pbp_team_game.csv'
    pbp=pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    rows=[]
    for season in [2023,2024,2025]:
        sg=raw[raw['season']==season]
        for week in sorted(pd.to_numeric(sg['week'],errors='coerce').dropna().astype(int).unique()):
            past=historico_antes(raw,season,week)
            pp=historico_antes(pbp,season,week) if not pbp.empty else pd.DataFrame()
            model=MoneylineRuntime()
            if not model.entrenar(past,pp): continue
            elo=MotorELONFL(); elo.actualizar_ratings(past)
            wk=sg[pd.to_numeric(sg['week'],errors='coerce')==week]
            for _,g in wk.iterrows():
                home,away=g.get('home_team'),g.get('away_team')
                if not home or not away or str(g.get('location','')).lower()=='neutral': continue
                hs,aws=g.get('home_score'),g.get('away_score')
                if pd.isna(hs) or pd.isna(aws): continue
                temp=pd.to_numeric(g.get('temp'),errors='coerce'); wind=pd.to_numeric(g.get('wind'),errors='coerce')
                dome=str(g.get('roof','')).lower() in {'dome','closed','indoor','indoors'}
                hr=pd.to_numeric(g.get('home_rest'),errors='coerce'); ar=pd.to_numeric(g.get('away_rest'),errors='coerce')
                pred=model.predecir_contexto(week,home,away,None if pd.isna(temp) else temp,None if pd.isna(wind) else wind,dome,None if pd.isna(hr) else hr,None if pd.isna(ar) else ar)
                if not pred: continue
                spread_line=pd.to_numeric(g.get('spread_line'),errors='coerce')
                total_line=pd.to_numeric(g.get('total_line'),errors='coerce')
                # nflverse spread_line: positivo = favorito local por esa cantidad.
                threshold=None if pd.isna(spread_line) else float(spread_line)
                emp=simular_nfl_montecarlo(home,away,past,None if pd.isna(total_line) else float(total_line),threshold)
                if not emp.get('Disponible'): continue
                ph,pa=empirical_residual_two_way(pred['ML_Margen_Local_Esperado'],0,model.residuales_margen)
                eh,ea=norm2(emp['Moneyline']['Gana Local'],emp['Moneyline']['Gana Visita'])
                elo_h=100*elo.calcular_probabilidad_elo(elo.ratings.get(home,1500),elo.ratings.get(away,1500))
                hm,am=g.get('home_moneyline'),g.get('away_moneyline')
                # ML reproduce el guardrail adicional Elo+MC.
                if not pd.isna(hm) and not pd.isna(am) and ph is not None and eh is not None:
                    if max(ph,elo_h,eh)-min(ph,elo_h,eh)<=MAX_DISAGREE and all((x>=50)==(ph>=50) for x in [elo_h,eh]):
                        add(rows,season,week,g.get('game_id'),'ML','H',ph,eh,hm,am,float(hs),float(aws))
                    if max(pa,100-elo_h,ea)-min(pa,100-elo_h,ea)<=MAX_DISAGREE and all((x>=50)==(pa>=50) for x in [100-elo_h,ea]):
                        add(rows,season,week,g.get('game_id'),'ML','A',pa,ea,am,hm,float(hs),float(aws))
                # Spread: convertir línea histórica nflverse al formato de apuesta.
                if threshold is not None:
                    sph,spa=empirical_residual_two_way(pred['ML_Margen_Local_Esperado'],threshold,model.residuales_margen)
                    mch=emp['Spread']['Cubre Local']; mca=emp['Spread']['Cubre Visita']
                    # nflverse conserva precios históricos home/away spread si existen.
                    sho=g.get('home_spread_odds'); sao=g.get('away_spread_odds')
                    if not pd.isna(sho) and not pd.isna(sao):
                        add(rows,season,week,g.get('game_id'),'SPREAD','H',sph,mch,sho,sao,float(hs),float(aws),-threshold)
                        add(rows,season,week,g.get('game_id'),'SPREAD','A',spa,mca,sao,sho,float(hs),float(aws),threshold)
                if not pd.isna(total_line):
                    po,pu=empirical_residual_two_way(pred['ML_Puntos_Totales_Esperados'],float(total_line),model.residuales_total)
                    mco=emp['Over_Under']['Prob Over']; mcu=emp['Over_Under']['Prob Under']
                    oo=g.get('over_odds'); uo=g.get('under_odds')
                    if not pd.isna(oo) and not pd.isna(uo):
                        add(rows,season,week,g.get('game_id'),'TOTAL','O',po,mco,oo,uo,float(hs),float(aws),float(total_line))
                        add(rows,season,week,g.get('game_id'),'TOTAL','U',pu,mcu,uo,oo,float(hs),float(aws),float(total_line))
    out=pd.DataFrame(rows)
    out.to_csv('backtest_nfl_production_all_markets_results.csv',index=False)
    if out.empty:
        raise SystemExit('No se generaron picks; revisar nombres de columnas históricas')
    print('\n=== NFL PRODUCTION ALL MARKETS WALK-FORWARD ===')
    for market,g in out.groupby('market'):
        print(market, {'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2),'avg_p':round(g.probability.mean(),2)})
    out['bin']=pd.cut(out.probability,[58,60,65,70,75,101],right=False,include_lowest=True)
    print('\nCALIBRATION BINS')
    print(out.groupby(['market','bin'],observed=True).agg(n=('win','size'),pred=('probability','mean'),actual=('win','mean'),roi=('return','mean')).to_string())
    assert set(out.market).issubset({'ML','SPREAD','TOTAL'})

if __name__=='__main__':
    main()
