from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class PredictorNFL_ML:
    """Predice total y margen usando únicamente información previa a la semana del juego."""

    def __init__(self):
        self.modelo_puntos_totales = RandomForestRegressor(
            n_estimators=250, max_depth=9, min_samples_leaf=6,
            random_state=42, n_jobs=1
        )
        self.modelo_margen = RandomForestRegressor(
            n_estimators=250, max_depth=9, min_samples_leaf=6,
            random_state=43, n_jobs=1
        )
        self.is_trained = False
        self.features_cols = []
        self.historial_actual = {}
        self.residuales_total = np.array([], dtype=float)
        self.residuales_margen = np.array([], dtype=float)
        self.altitud_estadios = {
            "DEN": 5280, "ARI": 1090, "ATL": 1050, "LV": 2000,
            "KC": 900, "DAL": 550, "GB": 700, "CAR": 750,
        }

    @staticmethod
    def _media(vals, n):
        vals = list(vals)[-n:]
        return float(np.mean(vals)) if vals else np.nan

    @staticmethod
    def _std(vals, n):
        vals = list(vals)[-n:]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan

    def _features_equipo(self, hist, team, prefijo):
        h = hist[team]
        return {
            f"{prefijo}_off_5": self._media(h["pf"], 5),
            f"{prefijo}_def_5": self._media(h["pa"], 5),
            f"{prefijo}_margin_5": self._media(h["margin"], 5),
            f"{prefijo}_total_5": self._media(h["total"], 5),
            f"{prefijo}_off_17": self._media(h["pf"], 17),
            f"{prefijo}_def_17": self._media(h["pa"], 17),
            f"{prefijo}_margin_17": self._media(h["margin"], 17),
            f"{prefijo}_total_17": self._media(h["total"], 17),
            f"{prefijo}_score_sd_17": self._std(h["pf"], 17),
        }

    @staticmethod
    def _nuevo_historial():
        return defaultdict(lambda: {
            "pf": deque(maxlen=34), "pa": deque(maxlen=34),
            "margin": deque(maxlen=34), "total": deque(maxlen=34),
        })

    @staticmethod
    def _actualizar_historial(hist, r):
        home, away = r.get("home_team"), r.get("away_team")
        if not home or not away or pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            return
        hs, aws = float(r["home_score"]), float(r["away_score"])
        total = hs + aws
        hist[home]["pf"].append(hs); hist[home]["pa"].append(aws)
        hist[home]["margin"].append(hs - aws); hist[home]["total"].append(total)
        hist[away]["pf"].append(aws); hist[away]["pa"].append(hs)
        hist[away]["margin"].append(aws - hs); hist[away]["total"].append(total)

    def construir_features_pregame(self, df_games):
        df = df_games.copy()
        if "game_type" in df.columns:
            df = df[df["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        sort_cols = [c for c in ["season", "week", "gameday", "game_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        hist = self._nuevo_historial()
        rows = []
        group_cols = [c for c in ["season", "week"] if c in df.columns]
        grupos = df.groupby(group_cols, sort=False) if len(group_cols) == 2 else [(None, df)]

        for _, semana_df in grupos:
            # Primero se construyen TODAS las features de la semana con el historial
            # disponible al cierre de la semana anterior. Solo después se cargan resultados.
            for _, r in semana_df.iterrows():
                home, away = r.get("home_team"), r.get("away_team")
                if not home or not away:
                    continue
                if len(hist[home]["pf"]) >= 4 and len(hist[away]["pf"]) >= 4:
                    temp_raw = pd.to_numeric(r.get("temp"), errors="coerce")
                    wind_raw = pd.to_numeric(r.get("wind"), errors="coerce")
                    row = {
                        "game_id": r.get("game_id"), "season": r.get("season"), "week": r.get("week"),
                        "home_team": home, "away_team": away,
                        "home_altitude": float(self.altitud_estadios.get(home, 0.0)),
                        "is_dome": 1 if str(r.get("roof", "")).lower() in {"dome", "closed"} else 0,
                        "temp": temp_raw, "wind": wind_raw,
                        "temp_missing": int(pd.isna(temp_raw)), "wind_missing": int(pd.isna(wind_raw)),
                        "home_rest": pd.to_numeric(r.get("home_rest"), errors="coerce"),
                        "away_rest": pd.to_numeric(r.get("away_rest"), errors="coerce"),
                        "puntos_totales": float(r["home_score"] + r["away_score"]),
                        "margen_local": float(r["home_score"] - r["away_score"]),
                    }
                    row.update(self._features_equipo(hist, home, "home"))
                    row.update(self._features_equipo(hist, away, "away"))
                    rows.append(row)

            for _, r in semana_df.iterrows():
                self._actualizar_historial(hist, r)

        self.historial_actual = hist
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        for c in ["temp", "wind", "home_rest", "away_rest"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    def entrenar(self, df_games, df_qbs=None):
        try:
            df = self.construir_features_pregame(df_games)
            if len(df) < 200:
                return False
            self.features_cols = [
                "week", "home_altitude", "temp", "wind", "is_dome", "temp_missing", "wind_missing",
                "home_rest", "away_rest",
                "home_off_5", "home_def_5", "home_margin_5", "home_total_5",
                "home_off_17", "home_def_17", "home_margin_17", "home_total_17", "home_score_sd_17",
                "away_off_5", "away_def_5", "away_margin_5", "away_total_5",
                "away_off_17", "away_def_17", "away_margin_17", "away_total_17", "away_score_sd_17",
            ]
            df = df.dropna(subset=[c for c in self.features_cols if c not in {"temp", "wind", "home_rest", "away_rest"}])
            X = df[self.features_cols]
            y_total, y_margin = df["puntos_totales"], df["margen_local"]

            cut = max(150, int(len(df) * 0.80))
            if cut < len(df) - 30:
                self.modelo_puntos_totales.fit(X.iloc[:cut], y_total.iloc[:cut])
                self.modelo_margen.fit(X.iloc[:cut], y_margin.iloc[:cut])
                self.residuales_total = (y_total.iloc[cut:] - self.modelo_puntos_totales.predict(X.iloc[cut:])).to_numpy()
                self.residuales_margen = (y_margin.iloc[cut:] - self.modelo_margen.predict(X.iloc[cut:])).to_numpy()

            self.modelo_puntos_totales.fit(X, y_total)
            self.modelo_margen.fit(X, y_margin)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando modelo ML NFL: {e}")
            self.is_trained = False
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
        datos = pd.DataFrame([row])[self.features_cols]
        if datos.isna().any(axis=None):
            return None
        total = float(self.modelo_puntos_totales.predict(datos)[0])
        margin = float(self.modelo_margen.predict(datos)[0])
        sigma_total = float(np.std(self.residuales_total, ddof=1)) if len(self.residuales_total) >= 20 else None
        sigma_margin = float(np.std(self.residuales_margen, ddof=1)) if len(self.residuales_margen) >= 20 else None
        return {
            "ML_Puntos_Totales_Esperados": round(total, 2),
            "ML_Margen_Local_Esperado": round(margin, 2),
            "Sigma_Total_OOS": None if sigma_total is None else round(sigma_total, 3),
            "Sigma_Margen_OOS": None if sigma_margin is None else round(sigma_margin, 3),
        }
