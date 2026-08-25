import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.nfl_ml_engine import PredictorNFL_ML


class PredictorNFL_PBPMax(PredictorNFL_ML):
    """Extensión PBP orientada a Moneyline con matchups explícitos.

    Mantiene el modelo base intacto. Usa únicamente features PBP rolling que ya
    fueron construidas con información disponible hasta la semana anterior.
    """

    def __init__(self):
        super().__init__()
        self.modelo_margen = RandomForestRegressor(
            n_estimators=350, max_depth=8, min_samples_leaf=8,
            max_features=0.70, random_state=73, n_jobs=1
        )
        self.modelo_moneyline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.20, max_iter=3000, class_weight=None, random_state=73)),
        ])
        self.moneyline_trained = False
        self.features_moneyline = []
        self.moneyline_oos_accuracy = None
        self.moneyline_oos_brier = None

    @staticmethod
    def _matchup_names():
        names = []
        for w in (4, 8):
            names += [
                f"home_epa_match_{w}", f"away_epa_match_{w}", f"epa_match_diff_{w}",
                f"home_success_match_{w}", f"away_success_match_{w}", f"success_match_diff_{w}",
                f"home_explosive_match_{w}", f"away_explosive_match_{w}", f"explosive_match_diff_{w}",
                f"home_pressure_edge_{w}", f"away_pressure_edge_{w}", f"pressure_edge_diff_{w}",
                f"home_pass_rush_gap_{w}", f"away_pass_rush_gap_{w}",
                f"pace_sum_{w}",
            ]
        return names

    @staticmethod
    def _add_matchups(frame):
        df = frame.copy()
        for w in (4, 8):
            h_epa = df[f"home_off_epa_play_{w}"] - df[f"away_def_epa_allowed_{w}"]
            a_epa = df[f"away_off_epa_play_{w}"] - df[f"home_def_epa_allowed_{w}"]
            h_suc = df[f"home_off_success_rate_{w}"] - df[f"away_def_success_allowed_{w}"]
            a_suc = df[f"away_off_success_rate_{w}"] - df[f"home_def_success_allowed_{w}"]
            h_exp = df[f"home_explosive_rate_{w}"] - df[f"away_def_explosive_allowed_{w}"]
            a_exp = df[f"away_explosive_rate_{w}"] - df[f"home_def_explosive_allowed_{w}"]
            h_pr = df[f"home_pressure_rate_{w}"] - df[f"away_sack_rate_allowed_{w}"]
            a_pr = df[f"away_pressure_rate_{w}"] - df[f"home_sack_rate_allowed_{w}"]

            df[f"home_epa_match_{w}"] = h_epa
            df[f"away_epa_match_{w}"] = a_epa
            df[f"epa_match_diff_{w}"] = h_epa - a_epa
            df[f"home_success_match_{w}"] = h_suc
            df[f"away_success_match_{w}"] = a_suc
            df[f"success_match_diff_{w}"] = h_suc - a_suc
            df[f"home_explosive_match_{w}"] = h_exp
            df[f"away_explosive_match_{w}"] = a_exp
            df[f"explosive_match_diff_{w}"] = h_exp - a_exp
            df[f"home_pressure_edge_{w}"] = h_pr
            df[f"away_pressure_edge_{w}"] = a_pr
            df[f"pressure_edge_diff_{w}"] = h_pr - a_pr
            df[f"home_pass_rush_gap_{w}"] = df[f"home_pass_epa_{w}"] - df[f"home_rush_epa_{w}"]
            df[f"away_pass_rush_gap_{w}"] = df[f"away_pass_epa_{w}"] - df[f"away_rush_epa_{w}"]
            df[f"pace_sum_{w}"] = df[f"home_plays_{w}"] + df[f"away_plays_{w}"]
        return df

    def construir_features_pregame(self, df_games, df_pbp_team_game=None):
        df = super().construir_features_pregame(df_games, df_pbp_team_game)
        raw_pbp = self._pbp_feature_names()
        if not df.empty and all(c in df.columns for c in raw_pbp):
            df = self._add_matchups(df)
        return df

    def entrenar(self, df_games, df_qbs=None, df_pbp_team_game=None):
        try:
            self.pbp_team_game = df_pbp_team_game.copy() if df_pbp_team_game is not None else pd.DataFrame()
            df = self.construir_features_pregame(df_games, self.pbp_team_game)
            if len(df) < 300:
                return False

            self.features_total = self._base_feature_names()
            raw_pbp = self._pbp_feature_names()
            matchup = self._matchup_names()
            self.usa_pbp = bool(not self.pbp_team_game.empty and all(c in df.columns for c in raw_pbp + matchup))
            if not self.usa_pbp:
                return False

            # El total se mantiene BASE: PBP no demostró valor O/U.
            self.features_margen = self.features_total + raw_pbp + matchup
            # Moneyline usa una selección más compacta para reducir overfit.
            selected_raw = []
            for w in (4, 8):
                for side in ("home", "away"):
                    for metric in ("off_epa_play", "off_success_rate", "pass_epa", "rush_epa", "explosive_rate", "def_epa_allowed", "def_success_allowed", "pressure_rate"):
                        selected_raw.append(f"{side}_{metric}_{w}")
            self.features_moneyline = self.features_total + selected_raw + matchup
            self.features_cols = self.features_margen

            total_df = df.dropna(subset=self.features_total + ["puntos_totales"])
            margin_df = df.dropna(subset=self.features_margen + ["margen_local"])
            ml_df = df.dropna(subset=self.features_moneyline + ["margen_local"])
            if min(len(total_df), len(margin_df), len(ml_df)) < 300:
                return False

            Xt, yt = total_df[self.features_total], total_df["puntos_totales"]
            Xm, ym = margin_df[self.features_margen], margin_df["margen_local"]
            Xc = ml_df[self.features_moneyline]
            yc = (ml_df["margen_local"] > 0).astype(int)

            cut_t = int(len(total_df) * 0.80)
            self.modelo_puntos_totales.fit(Xt.iloc[:cut_t], yt.iloc[:cut_t])
            self.residuales_total = (yt.iloc[cut_t:] - self.modelo_puntos_totales.predict(Xt.iloc[cut_t:])).to_numpy()

            cut_m = int(len(margin_df) * 0.80)
            self.modelo_margen.fit(Xm.iloc[:cut_m], ym.iloc[:cut_m])
            self.residuales_margen = (ym.iloc[cut_m:] - self.modelo_margen.predict(Xm.iloc[cut_m:])).to_numpy()

            cut_c = int(len(ml_df) * 0.80)
            self.modelo_moneyline.fit(Xc.iloc[:cut_c], yc.iloc[:cut_c])
            if cut_c < len(ml_df) - 20:
                p = self.modelo_moneyline.predict_proba(Xc.iloc[cut_c:])[:, 1]
                y = yc.iloc[cut_c:].to_numpy()
                self.moneyline_oos_accuracy = float(np.mean((p >= 0.5) == y))
                self.moneyline_oos_brier = float(np.mean((p - y) ** 2))

            self.modelo_puntos_totales.fit(Xt, yt)
            self.modelo_margen.fit(Xm, ym)
            self.modelo_moneyline.fit(Xc, yc)
            self.moneyline_trained = True
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando PBP-Max: {e}")
            self.is_trained = False
            self.moneyline_trained = False
            return False

    def predecir_contexto(self, week, home_team, away_team, temp, wind, is_dome, home_rest=None, away_rest=None):
        if not self.is_trained or home_team not in self.historial_actual or away_team not in self.historial_actual:
            return None
        if len(self.historial_actual[home_team]["pf"]) < 4 or len(self.historial_actual[away_team]["pf"]) < 4:
            return None

        temp_val = pd.to_numeric(temp, errors="coerce")
        wind_val = pd.to_numeric(wind, errors="coerce")
        row = {
            "week": float(week), "home_altitude": float(self.altitud_estadios.get(home_team, 0.0)),
            "temp": 0.0 if pd.isna(temp_val) else float(temp_val),
            "wind": 0.0 if pd.isna(wind_val) else float(wind_val),
            "is_dome": int(bool(is_dome)), "temp_missing": int(pd.isna(temp_val)), "wind_missing": int(pd.isna(wind_val)),
            "home_rest": 0.0 if home_rest is None or pd.isna(home_rest) else float(home_rest),
            "away_rest": 0.0 if away_rest is None or pd.isna(away_rest) else float(away_rest),
        }
        row.update(self._features_equipo(self.historial_actual, home_team, "home"))
        row.update(self._features_equipo(self.historial_actual, away_team, "away"))
        from modules.nfl_pbp_engine import features_pbp_actuales
        pbp_now = features_pbp_actuales(self.pbp_team_game, home_team, away_team)
        if pbp_now is None:
            return None
        row.update(pbp_now)
        datos = self._add_matchups(pd.DataFrame([row]))

        total_x = datos[self.features_total]
        margin_x = datos[self.features_margen]
        money_x = datos[self.features_moneyline]
        if total_x.isna().any(axis=None) or margin_x.isna().any(axis=None) or money_x.isna().any(axis=None):
            return None

        total = float(self.modelo_puntos_totales.predict(total_x)[0])
        margin = float(self.modelo_margen.predict(margin_x)[0])
        p_home = float(self.modelo_moneyline.predict_proba(money_x)[0, 1] * 100.0)
        sigma_total = float(np.std(self.residuales_total, ddof=1)) if len(self.residuales_total) >= 20 else None
        sigma_margin = float(np.std(self.residuales_margen, ddof=1)) if len(self.residuales_margen) >= 20 else None
        return {
            "ML_Puntos_Totales_Esperados": round(total, 2),
            "ML_Margen_Local_Esperado": round(margin, 2),
            "Probabilidad_Local_ML": round(p_home, 2),
            "Probabilidad_Visita_ML": round(100.0 - p_home, 2),
            "Sigma_Total_OOS": None if sigma_total is None else round(sigma_total, 3),
            "Sigma_Margen_OOS": None if sigma_margin is None else round(sigma_margin, 3),
            "Moneyline_OOS_Accuracy": self.moneyline_oos_accuracy,
            "Moneyline_OOS_Brier": self.moneyline_oos_brier,
            "Usa_PBP_Real": True,
            "PBP_Aplicado_A": "Margen/Moneyline + matchups EPA/success/presion",
        }
