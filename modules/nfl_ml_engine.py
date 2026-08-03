import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PredictorNFL_ML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.modelo_puntos_totales = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.modelo_margen = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.is_trained = False
        
        # Mapa de altitud de estadios (en pies) - Factor clave en la NFL
        self.altitud_estadios = {
            'DEN': 5280, 'ARI': 1090, 'ATL': 1050, 'LV': 2000, 
            'KC': 900, 'DAL': 550, 'GB': 700, 'CAR': 750
            # Los demás estadios están prácticamente a nivel del mar (0-100 ft)
        }

    def _limpiar_y_preparar(self, df_games, df_qbs=None):
        """Prepara las variables complejas (Clima, Altitud, QB) para el entrenamiento"""
        df = df_games.copy()
        
        # 1. Asignar Altitud
        df['home_altitude'] = df['home_team'].map(self.altitud_estadios).fillna(50)
        
        # 2. Interpretar el Clima (Viento y Temperatura)
        # Convertir techos a formato binario (Dome = 1, Abierto = 0)
        df['is_dome'] = df['roof'].apply(lambda x: 1 if str(x).lower() in ['dome', 'closed'] else 0)
        
        # 3. Targets (Lo que queremos predecir)
        df['margen_local'] = df['home_score'] - df['away_score'] # Para el Spread
        df['puntos_totales'] = df['home_score'] + df['away_score'] # Para el Over/Under
        
        # Nos quedamos con los factores de contexto (Features)
        features = ['week', 'home_altitude', 'temp', 'wind', 'is_dome']
        
        # Nota: Aquí en el futuro uniremos df_qbs usando un merge para agregar
        # variables como 'home_qb_rating' y 'away_qb_rating'.
        
        # Limpiar valores nulos para el modelo
        df = df.dropna(subset=features + ['margen_local', 'puntos_totales'])
        return df, features

    def entrenar(self, df_games, df_qbs=None):
        try:
            df_listo, self.features_cols = self._limpiar_y_preparar(df_games, df_qbs)
            
            X = df_listo[self.features_cols]
            y_puntos = df_listo['puntos_totales']
            y_margen = df_listo['margen_local']
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenamos los cerebros de la IA
            self.modelo_puntos_totales.fit(X_scaled, y_puntos)
            self.modelo_margen.fit(X_scaled, y_margen)
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando modelo ML NFL: {e}")
            return False

    def predecir_contexto(self, week, home_team, temp, wind, is_dome, qb_rating_local=90, qb_rating_visita=90):
        """
        Genera una predicción basándose EXCLUSIVAMENTE en el entorno y contexto del juego actual.
        """
        if not self.is_trained:
            return None
            
        altitud = self.altitud_estadios.get(home_team, 50)
        
        # Crear la fila con las condiciones del partido de hoy
        datos_hoy = pd.DataFrame([{
            'week': week,
            'home_altitude': altitud,
            'temp': temp,
            'wind': wind,
            'is_dome': is_dome
        }])
        
        datos_scaled = self.scaler.transform(datos_hoy)
        
        pred_puntos = self.modelo_puntos_totales.predict(datos_scaled)[0]
        pred_margen = self.modelo_margen.predict(datos_scaled)[0]
        
        return {
            "ML_Puntos_Totales_Esperados": round(pred_puntos, 1),
            "ML_Margen_Local_Esperado": round(pred_margen, 1)
        }
