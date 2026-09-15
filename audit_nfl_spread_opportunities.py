"""Auditoria walk-forward de TODAS las oportunidades de spread.

No cambia produccion. Registra ambos lados antes del resultado y explica exactamente
que filtros aceptan/rechazan cada candidato. 2026 queda excluido de la validacion.
"""
import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from modules.nfl_calibration import empirical_residual_two_way, historico_antes
from modules.nfl_moneyline_runtime import MoneylineRuntime
from modules.nfl_montecarlo_sim import simular_nfl_montecarlo

MIN_P=58.0; MIN_MC=58.0; MIN_EDGE=3.0; MIN_EV=3.0; MAX_DISAGREE=15.0

def dec(am):
    try:
        x=float(am)
        if pd.isna(x) or x==0:return None
        return 1+x/100 if x>0 else 1+100/abs(x)
    except Exception:return None

def no_vig(a,b):
    da,db=dec(a),dec(b)
    if da is None or db is None:return None,None
    ia,ib=1/da,1/db;s=ia+ib
    return (ia/s,ib/s) if s else (None,None)

def evaluate(p,mc,odd,other):
    reasons=[]
    if p is None: reasons.append('NO_PRIMARY')
    if mc is None: reasons.append('NO_MC')
    if reasons:return None,None,None,False,'|'.join(reasons)
    p=float(p);mc=float(mc);m,_=no_vig(odd,other);d=dec(odd)
    if m is None or d is None:return None,None,None,False,'NO_PRICE'
    edge=(p/100-m)*100;ev=(p/100*d-1)*100
    if (p>=50)!=(mc>=50):reasons.append('SIDE_DISAGREE')
    if abs(p-mc)>MAX_DISAGREE:reasons.append('DISAGREE_GT15')
    if p<MIN_P:reasons.append('P_LT58')
    if mc<MIN_MC:reasons.append('MC_LT58')
    if edge<MIN_EDGE:reasons.append('EDGE_LT3')
    if ev<MIN_EV:reasons.append('EV_LT3')
    return edge,ev,d,not reasons,'PASS' if not reasons else '|'.join(reasons)

def grade(side,hs,aws,line):
    margin=hs-aws;adj=(margin+line) if side=='H' else (-margin+line)
    if abs(adj)<1e-9:return None
    return int(adj>0)

def bucket(line):
    a=abs(float(line))
    if a<=2.5:return '0-2.5'
    if a<=5.5:return '3-5.5'
    if a<=8.5:return '6-8.5'
    return '9+'

def main():
    raw=nfl.import_schedules([2021,2022,2023,2024,2025])
    raw=raw[raw['result'].notna()].copy()
    if 'game_type' in raw.columns:raw=raw[raw['game_type'].isin(['REG','POST','WC','DIV','CON','SB'])].copy()
    pbp_path='data/historico_nfl_pbp_team_game.csv';pbp=pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()
    rows=[]
    for season in [2023,2024,2025]:
      sg=raw[raw.season==season]
      for week in sorted(pd.to_numeric(sg.week,errors='coerce').dropna().astype(int).unique()):
        past=historico_antes(raw,season,week);pp=historico_antes(pbp,season,week) if not pbp.empty else pd.DataFrame()
        model=MoneylineRuntime()
        if not model.entrenar(past,pp):continue
        for _,g in sg[pd.to_numeric(sg.week,errors='coerce')==week].iterrows():
          home,away=g.get('home_team'),g.get('away_team')
          if not home or not away or str(g.get('location','')).lower()=='neutral':continue
          hs,aws=g.get('home_score'),g.get('away_score');threshold=pd.to_numeric(g.get('spread_line'),errors='coerce')
          sho,sao=g.get('home_spread_odds'),g.get('away_spread_odds')
          if any(pd.isna(x) for x in [hs,aws,threshold,sho,sao]):continue
          temp=pd.to_numeric(g.get('temp'),errors='coerce');wind=pd.to_numeric(g.get('wind'),errors='coerce')
          hr=pd.to_numeric(g.get('home_rest'),errors='coerce');ar=pd.to_numeric(g.get('away_rest'),errors='coerce')
          dome=str(g.get('roof','')).lower() in {'dome','closed','indoor','indoors'}
          pred=model.predecir_contexto(week,home,away,None if pd.isna(temp) else temp,None if pd.isna(wind) else wind,dome,None if pd.isna(hr) else hr,None if pd.isna(ar) else ar)
          if not pred:continue
          threshold=float(threshold);emp=simular_nfl_montecarlo(home,away,past,None,threshold)
          if not emp.get('Disponible'):continue
          ph,pa=empirical_residual_two_way(pred['ML_Margen_Local_Esperado'],threshold,model.residuales_margen)
          mch=emp['Spread']['Cubre Local'];mca=emp['Spread']['Cubre Visita']
          for side,team,line,p,mc,odd,other in [('H',home,-threshold,ph,mch,sho,sao),('A',away,threshold,pa,mca,sao,sho)]:
            edge,ev,d,accepted,reason=evaluate(p,mc,odd,other);win=grade(side,float(hs),float(aws),line)
            role='FAVORITE' if line<0 else ('UNDERDOG' if line>0 else 'PICKEM')
            rows.append({'season':season,'week':week,'game_id':g.get('game_id'),'home':home,'away':away,'team':team,'side':side,'line':line,'line_bucket':bucket(line),'role':role,'primary_p':p,'mc_p':mc,'p_minus_mc':None if p is None or mc is None else float(p)-float(mc),'odds':odd,'edge':edge,'ev':ev,'production_accept':accepted,'reject_reason':reason,'win':win,'return':None if win is None or d is None else (d-1 if win else -1.0),'pred_margin':pred['ML_Margen_Local_Esperado'],'raw_margin':pred.get('ML_Margen_Local_Raw'),'cal_slope':pred.get('Margen_Calibration_Slope'),'cal_intercept':pred.get('Margen_Calibration_Intercept')})
    out=pd.DataFrame(rows);out.to_csv('audit_nfl_spread_opportunities.csv',index=False)
    if out.empty:raise SystemExit('No spread opportunities')
    print('\n=== ALL SPREAD OPPORTUNITIES ===');print({'n':len(out),'games':out.game_id.nunique()})
    accepted=out[out.production_accept & out.win.notna()].copy()
    print('\n=== PRODUCTION ACCEPTED ===')
    for role,g in accepted.groupby('role'):
      print(role,{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2),'avg_p':round(g.primary_p.mean(),2),'avg_mc':round(g.mc_p.mean(),2)})
    print('\n=== REJECTION REASONS BY ROLE ===')
    print(out[~out.production_accept].groupby(['role','reject_reason']).size().sort_values(ascending=False).head(30).to_string())
    print('\n=== LINE BUCKETS ACCEPTED ===')
    if not accepted.empty:print(accepted.groupby(['role','line_bucket'],observed=True).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean'),avg_p=('primary_p','mean'),avg_mc=('mc_p','mean')).to_string())
    print('\n=== PRIMARY/MC CORRELATION ===')
    z=out[['primary_p','mc_p']].dropna();print(round(z.corr().iloc[0,1],4) if len(z)>2 else None)
    print('\n=== MC VETO ON PRIMARY-QUALIFIED ===')
    q=out[(out.primary_p>=MIN_P)&(out.edge>=MIN_EDGE)&(out.ev>=MIN_EV)]
    print(q.groupby('role').agg(n=('team','size'),mc_pass=('production_accept','sum'),avg_p=('primary_p','mean'),avg_mc=('mc_p','mean')).to_string())

if __name__=='__main__':main()
