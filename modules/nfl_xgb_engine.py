import numpy as np
import pandas as pd
import xgboost as xgb

from modules.nfl_ml_engine import PredictorNFL_ML


class PredictorXGBoostSpread:
    """Modelo ATS usando líneas reales y features disponibles prepartido."""

    def __init__(self):
        self.modelo = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        self.is_trained = False
        self.features = []
        self.builder = PredictorNFL_ML()
        self.oos_accuracy = None
        self.oos_brier = None

    def _preparar_datos_entrenamiento(self, df_games):
        required = {"spread_line", "total_line", "home_score", "away_score", "game_id"}
        if not required.issubset(df_games.columns):
            return pd.DataFrame(), []

        rolling = self.builder.construir_features_pregame(df_games)
        if rolling.empty:
            return pd.DataFrame(), []

        mercado = df_games[["game_id", "spread_line", "total_line"]].copy()
        df = rolling.merge(mercado, on="game_id", how="left")
        df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")
        df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce")

        # Push no es derrota ni victoria ATS: se excluye del target binario.
        adjusted = df["margen_local"] + df["spread_line"]
        df = df[adjusted.notna() & (adjusted != 0)].copy()
        df["cubre_spread_home"] = (df["margen_local"] + df["spread_line"] > 0).astype(int)

        self.features = [
            "week", "spread_line", "total_line",
            "home_margin_5", "away_margin_5",
            "home_margin_17", "away_margin_17",
            "home_off_5", "home_def_5", "away_off_5", "away_def_5",
            "home_rest", "away_rest",
        ]
        df = df.dropna(subset=self.features + ["cubre_spread_home"])
        return df, self.features

    def entrenar(self, df_games):
        try:
            df, self.features = self._preparar_datos_entrenamiento(df_games)
            if len(df) < 300:
                self.is_trained = False
                return False

            X = df[self.features]
            y = df["cubre_spread_home"]
            cut = int(len(df) * 0.80)
            X_train, X_test = X.iloc[:cut], X.iloc[cut:]
            y_train, y_test = y.iloc[:cut], y.iloc[cut:]

            self.modelo.fit(X_train, y_train)
            if len(X_test):
                p = self.modelo.predict_proba(X_test)[:, 1]
                pred = (p >= 0.5).astype(int)
                self.oos_accuracy = float(np.mean(pred == y_test.to_numpy()))
                self.oos_brier = float(np.mean((p - y_test.to_numpy()) ** 2))

            self.modelo.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando XGBoost NFL: {e}")
            self.is_trained = False
            return False

    def predecir_probabilidad_cover(self, week, spread_line, total_line, home_team=None, away_team=None, home_rest=None, away_rest=None):
        if not self.is_trained or spread_line is None or total_line is None:
            return None
        if not home_team or not away_team:
            return None
        hist = self.builder.historial_actual
        if home_team not in hist or away_team not in hist:
            return None

        try:
            home = self.builder._features_equipo(hist, home_team, "home")
            away = self.builder._features_equipo(hist, away_team, "away")
            row = {
                "week": float(week),
                "spread_line": float(spread_line),
                "total_line": float(total_line),
                "home_rest": 0.0 if home_rest is None or pd.isna(home_rest) else float(home_rest),
                "away_rest": 0.0 if away_rest is None or pd.isna(away_rest) else float(away_rest),
            }
            row.update(home)
            row.update(away)
            datos = pd.DataFrame([row])[self.features]
            if datos.isna().any(axis=None):
                return None
            prob = float(self.modelo.predict_proba(datos)[0][1]) * 100.0
            return round(prob, 2)
        except Exception:
            return None
