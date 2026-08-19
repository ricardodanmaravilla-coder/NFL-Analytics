import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

class PredictorXGBoostSpread:
    def __init__(self):
        # Clasificador XGBoost optimizado para evitar sobreajuste
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
        df = df_games.copy()
        
        df['spread_line'] = df['spread_line'].fillna(-3.0)
        df['total_line'] = df['total'].fillna(45.5) # FIX: Usamos la línea O/U previa al juego
        df['home_score'] = df['home_score'].fillna(21)
        df['away_score'] = df['away_score'].fillna(21)
        
        # Target Binario: 1 si cubre, 0 si no
        df['margen_real'] = df['home_score'] - df['away_score']
        df['cubre_spread_home'] = (df['margen_real'] + df['spread_line'] > 0).astype(int)
        
        df['week'] = df['week'].fillna(1)
        
        # FIX DATA LEAKAGE: Solo entrenamos con variables pre-partido
        self.features = ['week', 'spread_line', 'total_line']
        
        df = df.dropna(subset=self.features + ['cubre_spread_home'])
        return df, self.features

    def entrenar(self, df_games):
        try:
            df_listo, self.features = self._preparar_datos_entrenamiento(df_games)
            if len(df_listo) < 50: return False
                
            X = df_listo[self.features]
            y = df_listo['cubre_spread_home']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.modelo.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error entrenando XGBoost: {e}")
            return False

    def predecir_probabilidad_cover(self, week, spread_line, total_line):
        if not self.is_trained: return 50.0
            
        try:
            datos_nuevos = pd.DataFrame([{
                'week': week,
                'spread_line': spread_line,
                'total_line': total_line
            }])
            
            probabilidades = self.modelo.predict_proba(datos_nuevos)
            prob_cover_home = float(probabilidades[0][1]) * 100.0
            return round(prob_cover_home, 1)
        except:
            return 50.0
