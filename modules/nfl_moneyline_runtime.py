import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modules.nfl_bigdata_store import cargar_pbp_preferente
from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_pbp_engine import features_pbp_actuales


class MoneylineRuntime:
    """Runtime de producción para margen/Moneyline y total de puntos.

    Las probabilidades se calibran con predicciones y residuales estrictamente OOS
    generados por bloques temporales expanding-window. Para margen se corrige además
    la contracción hacia cero típica del Random Forest mediante una calibración afín
    aprendida exclusivamente sobre esas predicciones OOS. Esto evita favorecer de
    forma estructural al underdog cuando el favorito tiene un spread exigente.
    """

    def __init__(self):
        self.base = PredictorNFL_ML()
        self.modelo_puntos_totales = self._new_total_model()
        self.modelo_margen = self._new_margin_model()
        self.features_total = []
        self.features_margen = []
        self.residuales_total = np.array([], dtype=float)
        self.residuales_margen = np.array([], dtype=float)
        self.margin_calibration_intercept = 0.0
        self.margin_calibration_slope = 1.0
        self.pbp_team_game = pd.DataFrame()
        self.usa_pbp = False
        self.is_trained = False

    @staticmethod
    def _new_total_model():
        return RandomForestRegressor(n_estimators=250, max_depth=9, min_samples_leaf=6,
                                     random_state=42, n_jobs=1)

    @staticmethod
    def _new_margin_model():
        return RandomForestRegressor(n_estimators=250, max_depth=9, min_samples_leaf=6,
                                     random_state=43, n_jobs=1)

    @staticmethod
    def _walkforward_oos(X, y, model_factory, min_train=150, block_size=64):
        """Predicciones/targets OOS cronológicos; nunca mezcla futuro en entrenamiento."""
        n = len(X)
        if n < min_train + 30:
            return np.array([], dtype=float), np.array([], dtype=float)
        preds, actuals = [], []
        start = int(min_train)
        while start < n:
            end = min(n, start + int(block_size))
            if end - start <= 0:
                break
            m = model_factory()
            m.fit(X.iloc[:start], y.iloc[:start])
            pred = np.asarray(m.predict(X.iloc[start:end]), dtype=float)
            actual = y.iloc[start:end].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(actual)
            preds.extend(pred[mask].tolist())
            actuals.extend(actual[mask].tolist())
            start = end
        return np.asarray(preds, dtype=float), np.asarray(actuals, dtype=float)

    @classmethod
    def _walkforward_residuals(cls, X, y, model_factory, min_train=150, block_size=64):
        pred, actual = cls._walkforward_oos(X, y, model_factory, min_train, block_size)
        return actual - pred

    @staticmethod
    def _fit_affine_oos(pred, actual):
        """Ajusta actual ~= intercept + slope*pred usando solo pares OOS."""
        pred = np.asarray(pred, dtype=float)
        actual = np.asarray(actual, dtype=float)
        mask = np.isfinite(pred) & np.isfinite(actual)
        pred, actual = pred[mask], actual[mask]
        if len(pred) < 30 or float(np.std(pred)) < 1e-9:
            return 0.0, 1.0
        slope, intercept = np.polyfit(pred, actual, 1)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return 0.0, 1.0
        # Guardrail amplio contra calibradores patológicos; no fuerza favoritos.
        slope = float(np.clip(slope, 0.5, 2.0))
        intercept = float(np.clip(intercept, -7.0, 7.0))
        return intercept, slope

    @staticmethod
    def _pbp_seguro_para_games(df_games, df_pbp_team_game=None):
        pbp = df_pbp_team_game.copy() if df_pbp_team_game is not None else pd.DataFrame()
        if pbp.empty:
            try:
                pbp = cargar_pbp_preferente()
            except Exception:
                pbp = pd.DataFrame()
        if pbp.empty or df_games is None or df_games.empty or "game_id" not in df_games.columns or "game_id" not in pbp.columns:
            return pbp
        allowed_ids = set(df_games["game_id"].dropna().astype(str))
        return pbp[pbp["game_id"].astype(str).isin(allowed_ids)].copy()

    def entrenar(self, df_games, df_pbp_team_game=None):
        self.pbp_team_game = self._pbp_seguro_para_games(df_games, df_pbp_team_game)
        self.base.pbp_team_game = self.pbp_team_game
        df = self.base.construir_features_pregame(df_games, self.pbp_team_game)
        if len(df) < 200:
            return False

        base_features = self.base._base_feature_names()
        pbp_features = self.base._pbp_feature_names()
        self.usa_pbp = bool(not self.pbp_team_game.empty and all(c in df.columns for c in pbp_features))
        self.features_total = base_features
        self.features_margen = base_features + (pbp_features if self.usa_pbp else [])

        total_df = df.dropna(subset=self.features_total + ["puntos_totales"])
        margin_df = df.dropna(subset=self.features_margen + ["margen_local"])
        if len(total_df) < 200 or len(margin_df) < 200:
            return False

        Xt = total_df[self.features_total]
        yt = total_df["puntos_totales"]
        Xm = margin_df[self.features_margen]
        ym = margin_df["margen_local"]

        total_pred_oos, total_actual_oos = self._walkforward_oos(Xt, yt, self._new_total_model)
        margin_pred_oos, margin_actual_oos = self._walkforward_oos(Xm, ym, self._new_margin_model)
        if len(total_pred_oos) < 30 or len(margin_pred_oos) < 30:
            return False

        self.residuales_total = total_actual_oos - total_pred_oos
        intercept, slope = self._fit_affine_oos(margin_pred_oos, margin_actual_oos)
        self.margin_calibration_intercept = intercept
        self.margin_calibration_slope = slope
        calibrated_margin_oos = intercept + slope * margin_pred_oos
        self.residuales_margen = margin_actual_oos - calibrated_margin_oos
        if len(self.residuales_total) < 30 or len(self.residuales_margen) < 30:
            return False

        self.modelo_puntos_totales = self._new_total_model()
        self.modelo_margen = self._new_margin_model()
        self.modelo_puntos_totales.fit(Xt, yt)
        self.modelo_margen.fit(Xm, ym)

        self.is_trained = True
        return True

    def predecir_contexto(self, week, home_team, away_team, temp, wind, is_dome,
                           home_rest=None, away_rest=None):
        hist = self.base.historial_actual
        if not self.is_trained or home_team not in hist or away_team not in hist:
            return None
        if len(hist[home_team]["pf"]) < 4 or len(hist[away_team]["pf"]) < 4:
            return None

        temp_val = pd.to_numeric(temp, errors="coerce")
        wind_val = pd.to_numeric(wind, errors="coerce")
        row = {
            "week": float(week),
            "home_altitude": float(self.base.altitud_estadios.get(home_team, 0.0)),
            "temp": 0.0 if pd.isna(temp_val) else float(temp_val),
            "wind": 0.0 if pd.isna(wind_val) else float(wind_val),
            "is_dome": int(bool(is_dome)),
            "temp_missing": int(pd.isna(temp_val)),
            "wind_missing": int(pd.isna(wind_val)),
            "home_rest": 0.0 if home_rest is None or pd.isna(home_rest) else float(home_rest),
            "away_rest": 0.0 if away_rest is None or pd.isna(away_rest) else float(away_rest),
        }
        row.update(self.base._features_equipo(hist, home_team, "home"))
        row.update(self.base._features_equipo(hist, away_team, "away"))

        if self.usa_pbp:
            pbp_now = features_pbp_actuales(self.pbp_team_game, home_team, away_team)
            if pbp_now is None:
                return None
            row.update(pbp_now)

        data = pd.DataFrame([row])
        Xt = data[self.features_total]
        Xm = data[self.features_margen]
        if Xt.isna().any(axis=None) or Xm.isna().any(axis=None):
            return None

        total = float(self.modelo_puntos_totales.predict(Xt)[0])
        raw_margin = float(self.modelo_margen.predict(Xm)[0])
        margin = self.margin_calibration_intercept + self.margin_calibration_slope * raw_margin
        sigma_total = float(np.std(self.residuales_total, ddof=1)) if len(self.residuales_total) >= 20 else None
        sigma_margin = float(np.std(self.residuales_margen, ddof=1)) if len(self.residuales_margen) >= 20 else None
        return {
            "ML_Puntos_Totales_Esperados": round(total, 2),
            "ML_Margen_Local_Esperado": round(float(margin), 2),
            "ML_Margen_Local_Raw": round(raw_margin, 2),
            "Margen_Calibration_Intercept": round(self.margin_calibration_intercept, 4),
            "Margen_Calibration_Slope": round(self.margin_calibration_slope, 4),
            "Sigma_Total_OOS": None if sigma_total is None else round(sigma_total, 3),
            "Sigma_Margen_OOS": None if sigma_margin is None else round(sigma_margin, 3),
            "Usa_PBP_Real": bool(self.usa_pbp),
            "PBP_Aplicado_A": "Margen/Moneyline" if self.usa_pbp else "No disponible",
            "Calibracion": "expanding_walkforward_oos+affine_margin_oos",
            "N_Residuos_Total": int(len(self.residuales_total)),
            "N_Residuos_Margen": int(len(self.residuales_margen)),
        }
