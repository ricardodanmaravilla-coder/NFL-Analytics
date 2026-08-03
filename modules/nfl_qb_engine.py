import pandas as pd
import numpy as np

class PredictorYardasQB:
    def __init__(self, df_qbs):
        self.df_qbs = df_qbs

    def proyectar_yardas_qb(self, nombre_qb, linea_yardas_las_vegas):
        """
        Analiza el promedio reciente de yardas del QB y calcula 
        la probabilidad de superar el Over/Under de su línea.
        """
        if self.df_qbs.empty:
            return {"error": "No hay datos de QBs cargados."}
            
        # Filtrar estadísticas del mariscal de campo
        stats_qb = self.df_qbs[self.df_qbs['player_name'].str.contains(nombre_qb, case=False, na=False)]
        
        if stats_qb.empty:
            return {"error": f"No se encontraron registros para el QB: {nombre_qb}"}
            
        # Tomar los últimos juegos registrados para ver su tendencia actual
        ultimos_juegos = stats_qb.tail(6)
        yardas_promedio = ultimos_juegos['passing_yards'].mean()
        desviacion_yardas = ultimos_juegos['passing_yards'].std()
        
        if pd.isna(desviacion_yardas) or desviacion_yardas == 0:
            desviacion_yardas = 50.0 # Varianza estándar por pase
            
        # Simulación Montecarlo específica para las yardas del QB (10,000 simulaciones)
        simulaciones_yardas = np.random.normal(loc=yardas_promedio, scale=desviacion_yardas, size=10000)
        simulaciones_yardas = np.maximum(0, simulaciones_yardas) # No hay yardas negativas
        
        prob_over = np.sum(simulaciones_yardas > linea_yardas_las_vegas) / 10000.0
        prob_under = 1.0 - prob_over
        
        return {
            "QB": nombre_qb,
            "Yardas_Promedio_Recientes": round(yardas_promedio, 1),
            "Linea_Las_Vegas": linea_yardas_las_vegas,
            "Prob_Over_Yardas": round(prob_over * 100, 2),
            "Prob_Under_Yardas": round(prob_under * 100, 2),
            "Recomendacion": "OVER de Yardas" if prob_over > 0.60 else ("UNDER de Yardas" if prob_under > 0.60 else "Sin valor claro")
        }
