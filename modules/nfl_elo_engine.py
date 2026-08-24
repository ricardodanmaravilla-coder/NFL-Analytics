import numpy as np
import pandas as pd


class MotorELONFL:
    def __init__(self, k_factor=20, home_advantage=48):
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.ratings = {}

    def inicializar_ratings(self, equipos):
        self.ratings = {eq: 1500.0 for eq in equipos}

    def calcular_probabilidad_elo(self, elo_local, elo_visita):
        diff = (float(elo_local) + self.home_advantage) - float(elo_visita)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def actualizar_ratings(self, df_games):
        df = df_games.copy()
        if "game_type" in df.columns:
            df = df[df["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

        equipos = sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna()))
        self.inicializar_ratings(equipos)

        sort_cols = [c for c in ["season", "week", "gameday", "game_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        temporada_actual = None
        for _, row in df.iterrows():
            season = row.get("season")
            if temporada_actual is not None and pd.notna(season) and season > temporada_actual:
                for eq in self.ratings:
                    self.ratings[eq] = self.ratings[eq] * 0.67 + 1500.0 * 0.33
            if pd.notna(season):
                temporada_actual = season

            home, away = row.get("home_team"), row.get("away_team")
            if home not in self.ratings or away not in self.ratings:
                continue
            hs, aws = float(row["home_score"]), float(row["away_score"])
            elo_home, elo_away = self.ratings[home], self.ratings[away]
            prob_home = self.calcular_probabilidad_elo(elo_home, elo_away)
            resultado = 1.0 if hs > aws else (0.0 if aws > hs else 0.5)

            mov = abs(hs - aws)
            mult_mov = np.log(max(mov, 1.0) + 1.0) * (2.2 / (abs(elo_home - elo_away) * 0.001 + 2.2))
            cambio = self.k_factor * mult_mov * (resultado - prob_home)
            self.ratings[home] += cambio
            self.ratings[away] -= cambio

        return self.ratings

    def obtener_power_ranking(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
