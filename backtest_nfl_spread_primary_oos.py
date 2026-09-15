"""Experimento walk-forward: Spread usa la probabilidad OOS del margen como señal primaria.

Monte Carlo se conserva en la salida como diagnóstico, pero NO confirma ni veta Spread.
Moneyline y Totales conservan exactamente la lógica del backtest de producción.
"""
import backtest_nfl_production_all_markets as bt


def add_spread_primary(rows, season, week, gid, market, side, p, mc, odd, other, hs, aws, line=0):
    if market != 'SPREAD':
        return bt._original_add(rows, season, week, gid, market, side, p, mc, odd, other, hs, aws, line)

    if p is None:
        return
    p = float(p)
    mkt, _ = bt.no_vig(odd, other)
    d = bt.dec(odd)
    if mkt is None or d is None:
        return
    edge = (p / 100.0 - mkt) * 100.0
    ev = (p / 100.0 * d - 1.0) * 100.0
    if p < bt.MIN_P or edge < bt.MIN_EDGE or ev < bt.MIN_EV:
        return

    win = bt.grade(market, side, hs, aws, line)
    if win is None:
        return
    role = 'FAVORITE' if float(line) < 0 else ('UNDERDOG' if float(line) > 0 else 'PICKEM')
    rows.append({
        'season': season, 'week': week, 'game_id': gid, 'market': market, 'side': side,
        'line': line, 'spread_role': role, 'probability': p, 'mc_probability': mc,
        'edge': edge, 'ev': ev, 'odds': odd, 'win': win,
        'return': d - 1 if win else -1.0,
    })


if __name__ == '__main__':
    bt._original_add = bt.add
    bt.add = add_spread_primary
    bt.main()
