import numpy as np
import pandas as pd

from modules.nfl_calibration import empirical_residual_gt, historico_antes, primary_with_agreement


def test_historico_antes_excludes_target_week_and_future():
    df = pd.DataFrame({
        "season": [2024, 2025, 2025, 2025, 2026],
        "week": [18, 1, 4, 5, 1],
        "value": [1, 2, 3, 4, 5],
    })
    out = historico_antes(df, 2025, 5)
    assert out["value"].tolist() == [1, 2, 3]
    assert not ((out["season"] == 2025) & (out["week"] >= 5)).any()
    assert not (out["season"] > 2025).any()


def test_empirical_calibration_is_monotonic_in_prediction():
    residuals = np.array([-10, -7, -4, -2, -1, 0, 1, 2, 4, 7, 10] * 5, dtype=float)
    low = empirical_residual_gt(-3, 0, residuals, min_n=30)
    high = empirical_residual_gt(3, 0, residuals, min_n=30)
    assert low is not None and high is not None
    assert high > low
    assert 0 < low < 100 and 0 < high < 100


def test_empirical_calibration_refuses_small_sample():
    assert empirical_residual_gt(3, 0, [1, -1, 2], min_n=30) is None


def test_support_models_are_guardrails_not_averaged():
    # La probabilidad final debe seguir siendo la primaria calibrada.
    assert primary_with_agreement(58.0, [55.0, 60.0]) == 58.0
    # Dirección opuesta => NO BET.
    assert primary_with_agreement(58.0, [49.0, 60.0]) is None
    # Desacuerdo extremo => NO BET.
    assert primary_with_agreement(58.0, [43.0, 60.0]) is None
