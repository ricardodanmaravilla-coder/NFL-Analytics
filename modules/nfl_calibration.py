import numpy as np
import pandas as pd


def historico_antes(df, season, week):
    """Devuelve solo información que podía conocerse antes del kickoff objetivo."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    if "season" not in df.columns or "week" not in df.columns:
        return df.copy()

    y = pd.to_numeric(df["season"], errors="coerce")
    w = pd.to_numeric(df["week"], errors="coerce")
    target_y, target_w = int(season), int(week)
    mask = (y < target_y) | ((y == target_y) & (w < target_w))
    return df.loc[mask].copy()


def _clean_residuals(residuals, min_n):
    if residuals is None:
        return None
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    return r if len(r) >= int(min_n) else None


def empirical_residual_gt(prediction, threshold, residuals, min_n=30, prior_strength=12.0):
    """P(Y>threshold) con residuales OOS, sin supuesto Normal."""
    if prediction is None or threshold is None:
        return None
    r = _clean_residuals(residuals, min_n)
    if r is None:
        return None
    hits = float(np.sum(float(prediction) + r > float(threshold)))
    alpha = beta = float(prior_strength) / 2.0
    p = (hits + alpha) / (len(r) + alpha + beta)
    return round(float(p) * 100.0, 2)


def empirical_residual_two_way(prediction, threshold, residuals, min_n=30, prior_strength=12.0):
    """Probabilidades condicionales arriba/abajo excluyendo pushes/ties.

    Para Moneyline evita el error de definir P(away)=1-P(home), que trataría un
    empate NFL como victoria visitante. El prior Beta simétrico estabiliza muestras.
    """
    if prediction is None or threshold is None:
        return None, None
    r = _clean_residuals(residuals, min_n)
    if r is None:
        return None, None
    delta = float(prediction) + r - float(threshold)
    above = float(np.sum(delta > 0))
    below = float(np.sum(delta < 0))
    nonpush = above + below
    if nonpush <= 0:
        return None, None
    alpha = beta = float(prior_strength) / 2.0
    denom = nonpush + alpha + beta
    p_above = (above + alpha) / denom
    p_below = (below + beta) / denom
    return round(p_above * 100.0, 2), round(p_below * 100.0, 2)


def primary_with_agreement(primary_prob, support_probs, max_disagreement=15.0):
    """Usa una sola probabilidad calibrada; auxiliares solo son guardrails."""
    if primary_prob is None:
        return None
    p = float(primary_prob)
    supports = [float(x) for x in support_probs if x is not None and np.isfinite(float(x))]
    if len(supports) < 2:
        return None

    all_probs = [p] + supports
    if max(all_probs) - min(all_probs) > float(max_disagreement):
        return None

    primary_side = p >= 50.0
    if any((x >= 50.0) != primary_side for x in supports):
        return None
    return p


def calibration_diagnostics(residuals):
    r = np.asarray(residuals if residuals is not None else [], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return {"n": int(len(r)), "bias": None, "mae": None, "rmse": None}
    return {
        "n": int(len(r)),
        "bias": round(float(np.mean(r)), 4),
        "mae": round(float(np.mean(np.abs(r))), 4),
        "rmse": round(float(np.sqrt(np.mean(r ** 2))), 4),
    }
