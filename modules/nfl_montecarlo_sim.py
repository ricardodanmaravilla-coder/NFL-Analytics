import numpy as np
import pandas as pd

def obtener_metricas_equipo(df, equipo, es_local, ultimos_n=17):
    """
    Filtra los datos para utilizar únicamente los últimos N juegos (1 temporada completa).
    Esto elimina el sesgo estadístico de años anteriores.
    """
    # Ordenar cronológicamente para asegurar que agarramos los más recientes
    if 'season' in df.columns and 'week' in df.columns:
        df_ordenado = df.sort_values(by=['season', 'week'])
    else:
        df_ordenado = df.copy()

    if es_local:
        juegos = df_ordenado[df_ordenado['home_team'] == equipo].tail(ultimos_n)
        p = juegos['home_score'].dropna()
        pr = juegos['away_score'].dropna()
    else:
        juegos = df_ordenado[df_ordenado['away_team'] == equipo].tail(ultimos_n)
        p = juegos['away_score'].dropna()
        pr = juegos['home_score'].dropna()
        
    if len(juegos) < 5:
        return {"anotados_mean": 21.0, "anotados_std": 7.0, "recibidos_mean": 21.0, "recibidos_std": 7.0}
        
    return {
        "anotados_mean": p.mean(), "anotados_std": p.std(),
        "recibidos_mean": pr.mean(), "recibidos_std": pr.std()
    }

def simular_nfl_montecarlo(local, visita, df_games, linea_ou=45.5, spread_local=-3.0, n_simulaciones=1000000):
    stats_local = obtener_metricas_equipo(df_games, local, True)
    stats_visita = obtener_metricas_equipo(df_games, visita, False)
    
    exp_puntos_local = (stats_local["anotados_mean"] + stats_visita["recibidos_mean"]) / 2.0
    std_puntos_local = (stats_local["anotados_std"] + stats_visita["recibidos_std"]) / 2.0
    
    exp_puntos_visita = (stats_visita["anotados_mean"] + stats_local["recibidos_mean"]) / 2.0
    std_puntos_visita = (stats_visita["anotados_std"] + stats_local["recibidos_std"]) / 2.0

    # 1. Simulación Normal (Fluida)
    sim_norm_loc = np.random.normal(loc=exp_puntos_local, scale=std_puntos_local, size=n_simulaciones)
    sim_norm_vis = np.random.normal(loc=exp_puntos_visita, scale=std_puntos_visita, size=n_simulaciones)
    
    # 2. Simulación Poisson (Eventos Discretos, Clave para la NFL)
    sim_pois_loc = np.random.poisson(lam=exp_puntos_local, size=n_simulaciones)
    sim_pois_vis = np.random.poisson(lam=exp_puntos_visita, size=n_simulaciones)
    
    # Blending (Fusión) de ambos mundos para perfección matemática
    sim_local = np.round((sim_norm_loc + sim_pois_loc) / 2.0)
    sim_visita = np.round((sim_norm_vis + sim_pois_vis) / 2.0)
    
    sim_local = np.maximum(0, sim_local)
    sim_visita = np.maximum(0, sim_visita)
    
    # Tiempo Extra
    empates = sim_local == sim_visita
    if np.any(empates):
        moneda = np.random.choice([3, -3], size=np.sum(empates))
        sim_local[empates] += np.where(moneda == 3, 3, 0)
        sim_visita[empates] += np.where(moneda == -3, 3, 0)
        
    puntos_totales = sim_local + sim_visita
    margen_local = sim_local - sim_visita

    prob_gana_local = np.sum(sim_local > sim_visita) / n_simulaciones
    prob_over = np.sum(puntos_totales > linea_ou) / n_simulaciones
    prob_under = np.sum(puntos_totales < linea_ou) / n_simulaciones
    
    prob_cubre_local = np.sum(margen_local > -spread_local) / n_simulaciones
    prob_cubre_visita = 1.0 - prob_cubre_local
    
    return {
        "Proyeccion_Score": {
            local: round(exp_puntos_local, 1), visita: round(exp_puntos_visita, 1),
            "Total_Proyectado": round(exp_puntos_local + exp_puntos_visita, 1)
        },
        "Moneyline": { "Gana Local": round(prob_gana_local * 100, 2), "Gana Visita": round((1-prob_gana_local) * 100, 2) },
        "Over_Under": { "Linea": linea_ou, "Prob Over": round(prob_over * 100, 2), "Prob Under": round(prob_under * 100, 2) },
        "Spread": { "Linea Local": spread_local, "Cubre Local": round(prob_cubre_local * 100, 2), "Cubre Visita": round(prob_cubre_visita * 100, 2) }
    }
