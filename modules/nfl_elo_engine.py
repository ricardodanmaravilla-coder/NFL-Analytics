import pandas as pd
import numpy as np

class MotorELONFL:
    def __init__(self, k_factor=20, home_advantage=48):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}

    def inicializar_ratings(self, equipos):
        for eq in equipos:
            self.ratings[eq] = 1500.0

    def calcular_probabilidad_elo(self, elo_local, elo_visita):
        diff = (elo_local + self.home_advantage) - elo_visita
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def actualizar_ratings(self, df_games):
        equipos = sorted(list(set(df_games['home_team'].dropna().unique()) | set(df_games['away_team'].dropna().unique())))
        self.inicializar_ratings(equipos)
        
        df_ordenado = df_games.sort_values(by=['season', 'week'])
        temporada_actual = None
        
        for _, row in df_ordenado.iterrows():
            temporada_juego = row['season']
            
            # FIX SESGO: Regresión a la media entre temporadas (33%)
            if temporada_actual is not None and temporada_juego > temporada_actual:
                for eq in self.ratings:
                    self.ratings[eq] = (self.ratings[eq] * 0.67) + (1500.0 * 0.33)
            temporada_actual = temporada_juego

            home = row['home_team']
            away = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']
            
            if pd.isna(home_score) or pd.isna(away_score) or home not in self.ratings or away not in self.ratings:
                continue
                
            elo_home = self.ratings[home]
            elo_away = self.ratings[away]
            prob_home = self.calcular_probabilidad_elo(elo_home, elo_away)
            
            if home_score > away_score: resultado_real = 1.0
            elif away_score > home_score: resultado_real = 0.0
            else: resultado_real = 0.5
                
            diff_puntos = abs(home_score - away_score)
            mult_mov = np.log(max(diff_puntos, 1) + 1) * (2.2 / (float(abs(elo_home - elo_away)) * 0.001 + 2.2))
            
            cambio = self.k_factor * mult_mov * (resultado_real - prob_home)
            self.ratings[home] += cambio
            self.ratings[away] -= cambio
            
        return self.ratings

    def obtener_power_ranking(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
