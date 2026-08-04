import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PredictorXGBoostSpread:
    def __init__(self):
        # Clasificador XGBoost optimizado para evitar sobreajuste (overfitting)
        self.modelo = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss'
        )
        self.is_trained = False
        self.features = []

    def _preparar_datos_entrenamiento(self, df_games):
        """Prepara las características históricas para aprender qué equipos cubren el spread"""
        df = df_games.copy()
        
        # Si el histórico no tiene una línea de spread guardada, usamos un estándar de 0
        if 'spread_line' not in df.columns:
            df['spread_line'] = -3.0
            
        df['spread_line'] = df['spread_line'].fillna(-3.0)
        df['home_score'] = df['home_score'].fillna(21)
        df['away_score'] = df['away_score'].fillna(21)
        
        # Margen real del partido (Local - Visita)
        df['margen_real'] = df['home_score'] - df['away_score']
        
        # Target Binario: 1 si el equipo local cubrió el spread histórico, 0 si no
        # Condición de cobertura: Margen real + Spread histórico > 0
        df['cubre_spread_home'] = (df['margen_real'] + df['spread_line'] > 0).astype(int)
        
        # Variables predictoras (Features)
        df['diff_score_hist'] = df['home_score'] - df['away_score'] # Historial básico
        df['week'] = df['week'].fillna(1)
        
        features = ['week', 'spread_line', 'diff_score_hist']
        
        df = df.dropna(subset=features + ['cubre_spread_home'])
        return df, features

    def entrenar(self, df_games):
        try:
            df_listo, self.features = self._preparar_datos_entrenamiento(df_games)
            if len(df_listo) < 50:
                return False
                
            X = df_listo[self.features]
            y = df_listo['cubre_spread_home']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.modelo.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando XGBoost: {e}")
            return False

    def predecir_probabilidad_cover(self, week, spread_line, diff_score_hist):
        """Retorna la probabilidad exacta (0 a 100%) de que el local cubra según XGBoost"""
        if not self.is_trained:
            return 50.0
            
        try:
            datos_nuevos = pd.DataFrame([{
                'week': week,
                'spread_line': spread_line,
                'diff_score_hist': diff_score_hist
            }])
            
            # predict_proba devuelve [prob_no_cubre, prob_cubre]
            probabilidades = self.modelo.predict_proba(datos_nuevos)
            prob_cover_home = float(probabilidades[0][1]) * 100.0
            return round(prob_cover_home, 1)
        except:
            return 50.0
