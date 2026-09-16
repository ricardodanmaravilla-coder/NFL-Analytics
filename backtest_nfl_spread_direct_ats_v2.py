"""ATS v2 temporal: conserva el clasificador directo y calibra cada rol SOLO con picks OOS anteriores.
Validacion 2023-2025; 2026 excluido. No toca produccion.
"""
import pandas as pd
import backtest_nfl_spread_direct_ats as v1

MIN_HISTORY=20
MIN_ROLE_WR=0.50


def main():
    # Genera el replay directo original si no existe.
    v1.main()
    df=pd.read_csv('backtest_nfl_spread_direct_ats_results.csv')
    df=df.sort_values(['season','week','game_id','side']).reset_index(drop=True)
    keep=[]
    history=[]
    for _,r in df.iterrows():
        past=pd.DataFrame(history)
        allow=True
        if not past.empty:
            rp=past[past.role==r.role]
            # El filtro se activa solo tras evidencia OOS suficiente y nunca mira el futuro.
            if len(rp)>=MIN_HISTORY and float(rp.win.mean()) < MIN_ROLE_WR:
                allow=False
        if allow: keep.append(r.to_dict())
        history.append(r.to_dict())
    out=pd.DataFrame(keep)
    out.to_csv('backtest_nfl_spread_direct_ats_v2_results.csv',index=False)
    print('\n=== DIRECT ATS V2 TEMPORAL ROLE GUARD ===')
    print({'n':len(out),'winrate':round(100*out.win.mean(),2),'roi':round(100*out['return'].mean(),2)})
    print('\nBY ROLE')
    for role,g in out.groupby('role'):
        print(role,{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2)})
    print('\nBY SEASON')
    for s,g in out.groupby('season'):
        print(int(s),{'n':len(g),'winrate':round(100*g.win.mean(),2),'roi':round(100*g['return'].mean(),2)})
    print('\nROLE x SEASON')
    print(out.groupby(['season','role']).agg(n=('win','size'),winrate=('win','mean'),roi=('return','mean')).to_string())

if __name__=='__main__': main()
