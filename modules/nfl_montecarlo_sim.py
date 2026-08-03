import numpy as np
import pandas as pd

def obtener_metricas_equipo(df, equipo, es_local):
    """
    Calcula el promedio y la volatilidad (desviación estándar) de puntos 
    anotados y recibidos de un equipo, separando si juega en casa o de visita.
    """
    if es_local:
        juegos = df[df['home_team'] == equipo]
        puntos_anotados = juegos['home_score'].dropna()
        puntos_recibidos = juegos['away_score'].dropna()
    else:
        juegos = df[df['away_team'] == equipo]
        puntos_anotados = juegos['away_score'].dropna()
        puntos_recibidos = juegos['home_score'].dropna()
        
    if len(juegos) < 5:
        # Si por alguna razón hay pocos datos (equipo nuevo/cambio de nombre), valores base NFL
        return {"anotados_mean": 21.0, "anotados_std": 7.0, "recibidos_mean": 21.0, "recibidos_std": 7.0}
        
    return {
        "anotados_mean": puntos_anotados.mean(),
        "anotados_std": puntos_anotados.std(),
        "recibidos_mean": puntos_recibidos.mean(),
        "recibidos_std": puntos_recibidos.std()
    }

def simular_nfl_montecarlo(local, visita, df_games, linea_ou=45.5, spread_local=-3.0, n_simulaciones=1000000):
    """
    Simula un partido de la NFL 1,000,000 veces cruzando el ataque de uno vs la defensa del otro.
    Retorna probabilidades de victoria, cobertura de spread (hándicap) y Over/Under.
    """
    # 1. Obtener métricas históricas de ambos equipos
    stats_local = obtener_metricas_equipo(df_games, local, es_local=True)
    stats_visita = obtener_metricas_equipo(df_games, visita, es_local=False)
    
    # Promedio de la liga para estabilizar (aproximadamente 22 puntos por equipo)
    promedio_liga = 22.0
    
    # 2. Calcular la expectativa real de puntos para este duelo específico
    # Fuerza de ataque local cruzada con debilidad de defensa visitante
    exp_puntos_local = (stats_local["anotados_mean"] + stats_visita["recibidos_mean"]) / 2.0
    std_puntos_local = (stats_local["anotados_std"] + stats_visita["recibidos_std"]) / 2.0
    
    exp_puntos_visita = (stats_visita["anotados_mean"] + stats_local["recibidos_mean"]) / 2.0
    std_puntos_visita = (stats_visita["anotados_std"] + stats_local["recibidos_std"]) / 2.0

    # 3. Iniciar simulaciones (Distribución Normal)
    sim_local = np.random.normal(loc=exp_puntos_local, scale=std_puntos_local, size=n_simulaciones)
    sim_visita = np.random.normal(loc=exp_puntos_visita, scale=std_puntos_visita, size=n_simulaciones)
    
    # En la NFL no hay puntos negativos, ajustamos a 0 si el modelo arroja negativos
    sim_local = np.maximum(0, np.round(sim_local))
    sim_visita = np.maximum(0, np.round(sim_visita))
    
    # Prevenir empates exactos (en NFL son rarísimos, < 0.5%)
    empates = sim_local == sim_visita
    if np.any(empates):
        # Lanzar una "moneda" en tiempo extra para romper el empate
        moneda = np.random.choice([3, -3], size=np.sum(empates)) # Gol de campo en OT
        sim_local[empates] += np.where(moneda == 3, 3, 0)
        sim_visita[empates] += np.where(moneda == -3, 3, 0)
        
    puntos_totales = sim_local + sim_visita
    margen_local = sim_local - sim_visita

    # 4. Calcular probabilidades
    # Moneyline (Ganador Directo)
    prob_gana_local = np.sum(sim_local > sim_visita) / n_simulaciones
    prob_gana_visita = np.sum(sim_visita > sim_local) / n_simulaciones
    
    # Over / Under
    prob_over = np.sum(puntos_totales > linea_ou) / n_simulaciones
    prob_under = np.sum(puntos_totales < linea_ou) / n_simulaciones
    
    # Spread (Hándicap)
    # Si el spread_local es -3.0, el local debe ganar por más de 3. (margen > 3)
    prob_cubre_local = np.sum(margen_local > abs(spread_local) if spread_local < 0 else margen_local > -spread_local) / n_simulaciones
    prob_cubre_visita = 1.0 - prob_cubre_local
    
    return {
        "Proyeccion_Score": {
            local: round(exp_puntos_local, 1),
            visita: round(exp_puntos_visita, 1),
            "Total_Proyectado": round(exp_puntos_local + exp_puntos_visita, 1)
        },
        "Moneyline": {
            "Gana Local": round(prob_gana_local * 100, 2),
            "Gana Visita": round(prob_gana_visita * 100, 2)
        },
        "Over_Under": {
            "Linea": linea_ou,
            "Prob Over": round(prob_over * 100, 2),
            "Prob Under": round(prob_under * 100, 2)
        },
        "Spread": {
            "Linea Local": spread_local,
            "Cubre Local": round(prob_cubre_local * 100, 2),
            "Cubre Visita": round(prob_cubre_visita * 100, 2)
        }
    }
