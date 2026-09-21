import numpy as np
import pandas as pd


def _ordenar(df):
    cols = [c for c in ["season", "week", "gameday", "game_id"] if c in df.columns]
    return df.sort_values(cols) if cols else df.copy()


def _muestras_equipo(df, equipo, ultimos_n=17, venue_n=8):
    """Marcadores previos únicos y márgenes observados por equipo.

    Cada partido entra una sola vez. Además de PF/PA se conserva el margen real
    (PF-PA), que permite construir el Spread sin el producto cartesiano de scores.
    """
    df = _ordenar(df)
    home = df[df["home_team"] == equipo].copy()
    away = df[df["away_team"] == equipo].copy()

    general = pd.concat([
        home.assign(
            pf=home["home_score"],
            pa=home["away_score"],
        )[["season", "week", "gameday", "game_id", "pf", "pa"]],
        away.assign(
            pf=away["away_score"],
            pa=away["home_score"],
        )[["season", "week", "gameday", "game_id", "pf", "pa"]],
    ], ignore_index=True)
    general = general.dropna(subset=["pf", "pa"])
    sort_cols = [c for c in ["season", "week", "gameday", "game_id"] if c in general.columns]
    if sort_cols:
        general = general.sort_values(sort_cols)
    general = general.tail(ultimos_n).copy()
    general["margin"] = general["pf"].astype(float) - general["pa"].astype(float)

    return {
        "general_pf": general["pf"].to_numpy(dtype=float),
        "general_pa": general["pa"].to_numpy(dtype=float),
        "general_margin": general["margin"].to_numpy(dtype=float),
        "home_n": int(len(home.dropna(subset=["home_score", "away_score"]).tail(venue_n))),
        "away_n": int(len(away.dropna(subset=["home_score", "away_score"]).tail(venue_n))),
    }


def _spread_distribution(h_margin, a_margin):
    """Distribución de margen local sin fabricar cruces de anotaciones.

    El margen histórico del visitante está expresado desde la perspectiva del
    visitante. Para llevarlo a perspectiva local se invierte su signo. Se unen
    ambas muestras reales con el mismo peso por partido; no se cruzan PF/PA ni
    scores entre sí. Así el tamaño efectivo de muestra es el número de partidos
    observados, no miles de pseudo-partidos derivados de un producto cartesiano.
    """
    hm = np.asarray(h_margin, dtype=float)
    am = np.asarray(a_margin, dtype=float)
    return np.concatenate([hm, -am])


def simular_nfl_montecarlo(local, visita, df_games, linea_ou=None, spread_local=None, n_simulaciones=None):
    """Distribución empírica determinista basada en partidos reales únicos.

    Moneyline y Total conservan el método previo para no modificar esos mercados.
    Spread usa únicamente márgenes reales observados y nunca el producto cartesiano
    de scores. `spread_local` es el umbral de margen local: +3 significa que el
    local debe ganar por más de 3 para cubrir. Producción convierte home_spread de
    la casa con threshold=-home_spread antes de llamar esta función.
    """
    if df_games is None or df_games.empty:
        return {"Disponible": False, "Motivo": "Sin histórico real"}

    df = df_games.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])].copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()]

    h = _muestras_equipo(df, local)
    a = _muestras_equipo(df, visita)
    if len(h["general_pf"]) < 5 or len(a["general_pf"]) < 5:
        return {"Disponible": False, "Motivo": "Menos de 5 juegos reales por equipo"}

    h_off = h["general_pf"]
    h_def = h["general_pa"]
    a_off = a["general_pf"]
    a_def = a["general_pa"]

    # Distribuciones empíricas sin producto cartesiano de pseudo-partidos.
    # Cada observación histórica aporta una sola vez a la muestra efectiva.
    score_h = np.concatenate([h_off, a_def]) / 2.0
    score_a = np.concatenate([a_off, h_def]) / 2.0
    margin_dist = _spread_distribution(h["general_margin"], a["general_margin"])
    total_dist = np.concatenate([h_off + h_def, a_off + a_def])

    n_margin = float(margin_dist.size)
    n_total = float(total_dist.size)
    p_home = float(np.sum(margin_dist > 0) / n_margin)
    p_away = float(np.sum(margin_dist < 0) / n_margin)
    p_tie = float(np.sum(margin_dist == 0) / n_margin)

    ou = {"Linea": linea_ou, "Prob Over": None, "Prob Under": None, "Prob Push": None}
    if linea_ou is not None and pd.notna(linea_ou):
        line = float(linea_ou)
        ou.update({
            "Prob Over": round(float(np.sum(total_dist > line) / n_total) * 100, 2),
            "Prob Under": round(float(np.sum(total_dist < line) / n_total) * 100, 2),
            "Prob Push": round(float(np.sum(total_dist == line) / n_total) * 100, 2),
        })

    spread_margin = _spread_distribution(h["general_margin"], a["general_margin"])
    spread_n = float(spread_margin.size)
    spread = {
        "Umbral margen local": spread_local,
        "Cubre Local": None,
        "Cubre Visita": None,
        "Push": None,
        "Muestra efectiva": int(spread_margin.size),
        "Metodo": "margenes reales emparejados sin producto cartesiano",
    }
    if spread_local is not None and pd.notna(spread_local):
        line = float(spread_local)
        adjusted = spread_margin - line
        spread.update({
            "Cubre Local": round(float(np.sum(adjusted > 0) / spread_n) * 100, 2),
            "Cubre Visita": round(float(np.sum(adjusted < 0) / spread_n) * 100, 2),
            "Push": round(float(np.sum(adjusted == 0) / spread_n) * 100, 2),
        })

    return {
        "Disponible": True,
        "Metodo": "Distribucion empirica sin productos cartesianos; muestras reales unicas",
        "Muestras_Local": int(len(h_off)),
        "Muestras_Visita": int(len(a_off)),
        "Venue_Local_N": h["home_n"],
        "Venue_Visita_N": a["away_n"],
        "Proyeccion_Score": {
            local: round(float(np.mean(score_h)), 2),
            visita: round(float(np.mean(score_a)), 2),
            "Total_Proyectado": round(float(np.mean(total_dist)), 2),
        },
        "Moneyline": {
            "Gana Local": round(p_home * 100, 2),
            "Gana Visita": round(p_away * 100, 2),
            "Empate": round(p_tie * 100, 2),
        },
        "Over_Under": ou,
        "Spread": spread,
    }
