import numpy as np
import pandas as pd


class PredictorYardasQB:
    def __init__(self, df_qbs):
        self.df_qbs = df_qbs.copy() if df_qbs is not None else pd.DataFrame()

    def proyectar_yardas_qb(self, nombre_qb, linea_yardas_las_vegas, ultimos_n=10):
        """Probabilidad empírica usando yardas reales recientes; sin datos sintéticos."""
        if self.df_qbs.empty or "player_name" not in self.df_qbs.columns:
            return {"error": "No hay datos reales de QBs cargados."}

        stats = self.df_qbs[self.df_qbs["player_name"].astype(str).str.contains(nombre_qb, case=False, na=False)].copy()
        if stats.empty or "passing_yards" not in stats.columns:
            return {"error": f"No se encontraron registros para el QB: {nombre_qb}"}

        sort_cols = [c for c in ["season", "week"] if c in stats.columns]
        if sort_cols:
            stats = stats.sort_values(sort_cols)
        yards = pd.to_numeric(stats["passing_yards"], errors="coerce").dropna().tail(int(ultimos_n))
        if len(yards) < 5:
            return {"error": "Se requieren al menos 5 juegos reales recientes para evaluar el prop."}

        line = float(linea_yardas_las_vegas)
        over = float((yards > line).mean())
        under = float((yards < line).mean())
        push = float((yards == line).mean())

        return {
            "QB": nombre_qb,
            "Muestra_Juegos": int(len(yards)),
            "Yardas_Promedio_Recientes": round(float(yards.mean()), 1),
            "Yardas_Mediana_Recientes": round(float(yards.median()), 1),
            "Linea_Las_Vegas": line,
            "Prob_Over_Yardas": round(over * 100, 2),
            "Prob_Under_Yardas": round(under * 100, 2),
            "Prob_Push": round(push * 100, 2),
            "Metodo": "Frecuencia empirica de juegos reales",
            "Recomendacion": "OVER" if over >= 0.60 else ("UNDER" if under >= 0.60 else "Sin señal histórica fuerte"),
        }
