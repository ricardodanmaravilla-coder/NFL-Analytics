import pandas as pd
import numpy as np

class MotorELONFL:
    def __init__(self, k_factor=20, home_advantage=48):
        self.k_factor = k_factor
        self.home_advantage = home_advantage # Puntos de ELO extra por jugar en casa
        self.ratings = {}

    def inicializar_ratings(self, equipos):
        # Todos los equipos arrancan con un ELO base estándar de la NFL (1500)
        for eq in equipos:
            self.ratings[eq] = 1500.0

    def calcular_probabilidad_elo(self, elo_local, elo_visita):
        # Probabilidad de victoria basada en la diferencia de ELO con factor de casa
        diff = (elo_local + self.home_advantage) - elo_visita
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def actualizar_ratings(self, df_games):
        """Procesa todo el histórico de partidos para actualizar el ELO de cada equipo cronológicamente"""
        equipos = sorted(list(set(df_games['home_team'].dropna().unique()) | set(df_games['away_team'].dropna().unique())))
        self.inicializar_ratings(equipos)
        
        # Ordenar cronológicamente por temporada y semana
        df_ordenado = df_games.sort_values(by=['season', 'week'])
        
        for _, row in df_ordenado.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']
            
            if pd.isna(home_score) or pd.isna(away_score) or home not in self.ratings or away not in self.ratings:
                continue
                
            elo_home = self.ratings[home]
            elo_away = self.ratings[away]
            
            # Probabilidad esperada
            prob_home = self.calcular_probabilidad_elo(elo_home, elo_away)
            
            # Resultado real (1 = Gana local, 0 = Gana visita, 0.5 = Empate/Raro en NFL)
            if home_score > away_score:
                resultado_real = 1.0
            elif away_score > home_score:
                resultado_real = 0.0
            else:
                resultado_real = 0.5
                
            # Ajuste por Margen de Victoria (MoV multiplier): En la NFL golear importa más
            diff_puntos = abs(home_score - away_score)
            mult_mov = np.log(max(diff_puntos, 1) + 1) * (2.2 / (float(abs(elo_home - elo_away)) * 0.001 + 2.2))
            
            # Actualizar ELO
            cambio = self.k_factor * mult_mov * (resultado_real - prob_home)
            self.ratings[home] += cambio
            self.ratings[away] -= cambio
            
        return self.ratings

    def obtener_power_ranking(self):
        """Devuelve los equipos ordenados por su ELO actual"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
