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
        
        # Diccionarios dinámicos para almacenar el poder ofensivo y defensivo real por equipo
        self.power_off_home = {}
        self.power_def_home = {}
        self.power_off_away = {}
        self.power_def_away = {}
        
        # Mapa de altitud de estadios (en pies)
        self.altitud_estadios = {
            'DEN': 5280, 'ARI': 1090, 'ATL': 1050, 'LV': 2000, 
            'KC': 900, 'DAL': 550, 'GB': 700, 'CAR': 750
        }

    def _calcular_poderes_unidades(self, df_games):
        """Calcula el poder ofensivo y defensivo real basado en el histórico de puntos"""
        home_off = df_games.groupby('home_team')['home_score'].mean().to_dict()
        home_def = df_games.groupby('home_team')['away_score'].mean().to_dict()
        away_off = df_games.groupby('away_team')['away_score'].mean().to_dict()
        away_def = df_games.groupby('away_team')['home_score'].mean().to_dict()
        
        # Combinar promedios globales por equipo (Local + Visita)
rm = set(list(home_off.keys()) + list(away_off.keys()))
        
        self.power_off = {}
        self.power_def = {}
        
        for eq in rm:
            pts_a_favor = []
            pts_en_contra = []
            if eq in home_off: pts_a_favor.append(home_off[eq])
            if eq in away_off: pts_a_favor.append(away_off[eq])
            if eq in home_def: pts_en_contra.append(home_def[eq])
            if eq in away_def: pts_en_contra.append(away_def[eq])
            
            self.power_off[eq] = np.mean(pts_a_favor) if pts_a_favor else 22.0
            self.power_def[eq] = np.mean(pts_en_contra) if pts_en_contra else 22.0

    def _limpiar_y_preparar(self, df_games, df_qbs=None):
        df = df_games.copy()
        
        # Calcular los poderes ofensivos y defensivos reales
        self._calcular_poderes_unidades(df)
        
        # Asignar métricas de unidades al DataFrame de entrenamiento
        df['home_offense_power'] = df['home_team'].map(self.power_off).fillna(22.0)
        df['home_defense_power'] = df['home_team'].map(self.power_def).fillna(22.0)
        df['away_offense_power'] = df['away_team'].map(self.power_off).fillna(22.0)
        df['away_defense_power'] = df['away_team'].map(self.power_def).fillna(22.0)
        
        # Variables contextuales de entorno
        df['home_altitude'] = df['home_team'].map(self.altitud_estadios).fillna(50)
        df['is_dome'] = df['roof'].apply(lambda x: 1 if str(x).lower() in ['dome', 'closed'] else 0)
        
        # Targets
        df['margen_local'] = df['home_score'] - df['away_score']
        df['puntos_totales'] = df['home_score'] + df['away_score']
        
        # Features completas incluyendo Enfrentamiento Ofensiva vs Defensiva Real
        features = [
            'week', 'home_altitude', 'temp', 'wind', 'is_dome',
            'home_offense_power', 'home_defense_power', 
            'away_offense_power', 'away_defense_power'
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
        """
        Genera una predicción enfrentando el poder ofensivo y defensivo real de ambos equipos 
        sumado a las condiciones del estadio y clima.
        """
        if not self.is_trained:
            return None
            
        altitud = self.altitud_estadios.get(home_team, 50)
        
        h_off = self.power_off.get(home_team, 22.0)
        h_def = self.power_def.get(home_team, 22.0)
        a_off = self.power_off.get(away_team, 22.0)
        a_def = self.power_def.get(away_team, 22.0)
        
        datos_hoy = pd.DataFrame([{
            'week': week,
            'home_altitude': altitud,
            'temp': temp,
            'wind': wind,
            'is_dome': is_dome,
            'home_offense_power': h_off,
            'home_defense_power': h_def,
            'away_offense_power': a_off,
            'away_defense_power': a_def
        }])
        
        datos_scaled = self.scaler.transform(datos_hoy)
        
        pred_puntos = self.modelo_puntos_totales.predict(datos_scaled)[0]
        pred_margen = self.modelo_margen.predict(datos_scaled)[0]
        
        return {
            "ML_Puntos_Totales_Esperados": round(pred_puntos, 1),
            "ML_Margen_Local_Esperado": round(pred_margen, 1)
        }
