import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PredictorNFL_ML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.modelo_puntos_totales = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.modelo_margen = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.is_trained = False
        
        self.power_off = {}
        self.power_def = {}
        self.pace = {}
        self.efficiency = {}
        
        self.altitud_estadios = {
            'DEN': 5280, 'ARI': 1090, 'ATL': 1050, 'LV': 2000, 
            'KC': 900, 'DAL': 550, 'GB': 700, 'CAR': 750
        }

    def _calcular_poderes_unidades(self, df_games, ultimos_n=17):
        """Calcula Poder, Ritmo (Pace) y Eficiencia Neta basado SOLO en los últimos 17 juegos por equipo"""
        if 'season' in df_games.columns and 'week' in df_games.columns:
            df_sorted = df_games.sort_values(by=['season', 'week'])
        else:
            df_sorted = df_games.copy()

        # Aislar los últimos N partidos para evitar el sesgo de 2020-2023
        home_off = df_sorted.groupby('home_team').tail(ultimos_n).groupby('home_team')['home_score'].mean().to_dict()
        home_def = df_sorted.groupby('home_team').tail(ultimos_n).groupby('home_team')['away_score'].mean().to_dict()
        away_off = df_sorted.groupby('away_team').tail(ultimos_n).groupby('away_team')['away_score'].mean().to_dict()
        away_def = df_sorted.groupby('away_team').tail(ultimos_n).groupby('away_team')['home_score'].mean().to_dict()
        
        # Ritmo de juego (Puntos totales combinados en sus últimos partidos)
        df_sorted['total_pts_game'] = df_sorted['home_score'] + df_sorted['away_score']
        pace_home = df_sorted.groupby('home_team').tail(ultimos_n).groupby('home_team')['total_pts_game'].mean().to_dict()
        pace_away = df_sorted.groupby('away_team').tail(ultimos_n).groupby('away_team')['total_pts_game'].mean().to_dict()
        
        teams = set(list(home_off.keys()) + list(away_off.keys()))
        
        self.power_off = {}
        self.power_def = {}
        self.pace = {}
        self.efficiency = {}
        
        for eq in teams:
            pts_a_favor = []
            pts_en_contra = []
            if eq in home_off: pts_a_favor.append(home_off[eq])
            if eq in away_off: pts_a_favor.append(away_off[eq])
            if eq in home_def: pts_en_contra.append(home_def[eq])
            if eq in away_def: pts_en_contra.append(away_def[eq])
            
            self.power_off[eq] = np.mean(pts_a_favor) if pts_a_favor else 22.0
            self.power_def[eq] = np.mean(pts_en_contra) if pts_en_contra else 22.0
            
            # Eficiencia (Puntos anotados vs recibidos)
            self.efficiency[eq] = self.power_off[eq] - self.power_def[eq]
            
            # Pace
            p_h = pace_home.get(eq, 44.0)
            p_a = pace_away.get(eq, 44.0)
            self.pace[eq] = (p_h + p_a) / 2.0

    def _limpiar_y_preparar(self, df_games, df_qbs=None):
        df = df_games.copy()
        self._calcular_poderes_unidades(df)
        
        df['home_offense_power'] = df['home_team'].map(self.power_off).fillna(22.0)
        df['home_defense_power'] = df['home_team'].map(self.power_def).fillna(22.0)
        df['away_offense_power'] = df['away_team'].map(self.power_off).fillna(22.0)
        df['away_defense_power'] = df['away_team'].map(self.power_def).fillna(22.0)
        
        df['home_efficiency'] = df['home_team'].map(self.efficiency).fillna(0.0)
        df['away_efficiency'] = df['away_team'].map(self.efficiency).fillna(0.0)
        df['home_pace'] = df['home_team'].map(self.pace).fillna(44.0)
        df['away_pace'] = df['away_team'].map(self.pace).fillna(44.0)
        
        df['home_altitude'] = df['home_team'].map(self.altitud_estadios).fillna(50)
        df['is_dome'] = df['roof'].apply(lambda x: 1 if str(x).lower() in ['dome', 'closed'] else 0)
        
        df['margen_local'] = df['home_score'] - df['away_score']
        df['puntos_totales'] = df['home_score'] + df['away_score']
        
        features = [
            'week', 'home_altitude', 'temp', 'wind', 'is_dome',
            'home_offense_power', 'home_defense_power', 
            'away_offense_power', 'away_defense_power',
            'home_efficiency', 'away_efficiency', 'home_pace', 'away_pace'
        ]
        
        df = df.dropna(subset=features + ['margen_local', 'puntos_totales'])
        return df, features

    def entrenar(self, df_games, df_qbs=None):
        try:
            df_listo, self.features_cols = self._limpiar_y_preparar(df_games, df_qbs)
            X = df_listo[self.features_cols]
            y_puntos = df_listo['puntos_totales']
            y_margen = df_listo['margen_local']
            
            X_scaled = self.scaler.fit_transform(X)
            self.modelo_puntos_totales.fit(X_scaled, y_puntos)
            self.modelo_margen.fit(X_scaled, y_margen)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando modelo ML NFL: {e}")
            return False

    def predecir_contexto(self, week, home_team, away_team, temp, wind, is_dome):
        if not self.is_trained:
            return None
            
        datos_hoy = pd.DataFrame([{
            'week': week,
            'home_altitude': self.altitud_estadios.get(home_team, 50),
            'temp': temp, 'wind': wind, 'is_dome': is_dome,
            'home_offense_power': self.power_off.get(home_team, 22.0),
            'home_defense_power': self.power_def.get(home_team, 22.0),
            'away_offense_power': self.power_off.get(away_team, 22.0),
            'away_defense_power': self.power_def.get(away_team, 22.0),
            'home_efficiency': self.efficiency.get(home_team, 0.0),
            'away_efficiency': self.efficiency.get(away_team, 0.0),
            'home_pace': self.pace.get(home_team, 44.0),
            'away_pace': self.pace.get(away_team, 44.0)
        }])
        
        datos_scaled = self.scaler.transform(datos_hoy)
        return {
            "ML_Puntos_Totales_Esperados": round(self.modelo_puntos_totales.predict(datos_scaled)[0], 1),
            "ML_Margen_Local_Esperado": round(self.modelo_margen.predict(datos_scaled)[0], 1)
        }
