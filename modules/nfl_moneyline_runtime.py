import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from modules.nfl_ml_engine import PredictorNFL_ML
from modules.nfl_pbp_engine import features_pbp_actuales


class MoneylineRuntime:
    """Memory-light runtime that trains only the validated margin/Moneyline forest.

    It deliberately skips the totals forest because the production scanner does not
    auto-bet totals. Feature construction, PBP inputs, RF hyperparameters and OOS
    residual calibration match PredictorNFL_ML's margin path.
    """

    def __init__(self):
        self.base = PredictorNFL_ML()
        self.modelo_margen = RandomForestRegressor(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=6,
            random_state=43,
            n_jobs=1,
        )
        self.features_margen = []
        self.residuales_margen = np.array([], dtype=float)
        self.pbp_team_game = pd.DataFrame()
        self.usa_pbp = False
        self.is_trained = False

    def entrenar(self, df_games, df_pbp_team_game=None):
        self.pbp_team_game = (
            df_pbp_team_game.copy() if df_pbp_team_game is not None else pd.DataFrame()
        )
        self.base.pbp_team_game = self.pbp_team_game
        df = self.base.construir_features_pregame(df_games, self.pbp_team_game)
        if len(df) < 200:
            return False

        base_features = self.base._base_feature_names()
        pbp_features = self.base._pbp_feature_names()
        self.usa_pbp = bool(
            not self.pbp_team_game.empty and all(c in df.columns for c in pbp_features)
        )
        self.features_margen = base_features + (pbp_features if self.usa_pbp else [])
        margin_df = df.dropna(subset=self.features_margen + ["margen_local"])
        if len(margin_df) < 200:
            return False

        X = margin_df[self.features_margen]
        y = margin_df["margen_local"]
        cut = max(150, int(len(margin_df) * 0.80))
        if cut < len(margin_df) - 30:
            self.modelo_margen.fit(X.iloc[:cut], y.iloc[:cut])
            self.residuales_margen = (
                y.iloc[cut:] - self.modelo_margen.predict(X.iloc[cut:])
            ).to_numpy()
        self.modelo_margen.fit(X, y)
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

        X = pd.DataFrame([row])[self.features_margen]
        if X.isna().any(axis=None):
            return None
        margin = float(self.modelo_margen.predict(X)[0])
        sigma = (
            float(np.std(self.residuales_margen, ddof=1))
            if len(self.residuales_margen) >= 20
            else None
        )
        return {
            "ML_Margen_Local_Esperado": round(margin, 2),
            "Sigma_Margen_OOS": None if sigma is None else round(sigma, 3),
            "Usa_PBP_Real": bool(self.usa_pbp),
            "PBP_Aplicado_A": "Margen/Moneyline" if self.usa_pbp else "No disponible",
        }
