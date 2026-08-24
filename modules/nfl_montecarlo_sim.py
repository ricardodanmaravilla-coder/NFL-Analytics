import numpy as np
import pandas as pd


def _ordenar(df):
    cols = [c for c in ["season", "week", "gameday", "game_id"] if c in df.columns]
    return df.sort_values(cols) if cols else df.copy()


def _muestras_equipo(df, equipo, ultimos_n=17, venue_n=8):
    """Devuelve únicamente marcadores reales previos del equipo.

    Se usan los últimos N juegos generales y, con peso adicional, los últimos juegos
    en la condición local/visita correspondiente. No se generan Normal/Poisson ni
    se rellenan muestras faltantes con promedios ficticios.
    """
    df = _ordenar(df)
    home = df[df["home_team"] == equipo].copy()
    away = df[df["away_team"] == equipo].copy()

    general = pd.concat([
        home.assign(pf=home["home_score"], pa=home["away_score"])[["season", "week", "pf", "pa"]],
        away.assign(pf=away["away_score"], pa=away["home_score"])[["season", "week", "pf", "pa"]],
    ], ignore_index=True)
    general = general.dropna(subset=["pf", "pa"]).sort_values(["season", "week"]).tail(ultimos_n)

    return {
        "general_pf": general["pf"].to_numpy(dtype=float),
        "general_pa": general["pa"].to_numpy(dtype=float),
        "home_pf": home.dropna(subset=["home_score"]).tail(venue_n)["home_score"].to_numpy(dtype=float),
        "home_pa": home.dropna(subset=["away_score"]).tail(venue_n)["away_score"].to_numpy(dtype=float),
        "away_pf": away.dropna(subset=["away_score"]).tail(venue_n)["away_score"].to_numpy(dtype=float),
        "away_pa": away.dropna(subset=["home_score"]).tail(venue_n)["home_score"].to_numpy(dtype=float),
    }


def _combinar_muestras(base, venue):
    base = np.asarray(base, dtype=float)
    venue = np.asarray(venue, dtype=float)
    if len(base) == 0:
        return np.array([], dtype=float)
    if len(venue) >= 3:
        return np.concatenate([base, venue])
    return base


def simular_nfl_montecarlo(local, visita, df_games, linea_ou=None, spread_local=None, n_simulaciones=None):
    """Distribución empírica determinista basada en marcadores reales.

    `spread_local` conserva la semántica nflverse: +3 significa que el local es
    favorito por 3; por ello cubre cuando margen_local > 3. No se usa RNG.
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

    h_off = _combinar_muestras(h["general_pf"], h["home_pf"])
    h_def = _combinar_muestras(h["general_pa"], h["home_pa"])
    a_off = _combinar_muestras(a["general_pf"], a["away_pf"])
    a_def = _combinar_muestras(a["general_pa"], a["away_pa"])

    score_h = ((h_off[:, None] + a_def[None, :]) / 2.0).reshape(-1)
    score_a = ((a_off[:, None] + h_def[None, :]) / 2.0).reshape(-1)

    sh = score_h[:, None]
    sa = score_a[None, :]
    total = sh + sa
    margin = sh - sa
    n = float(total.size)

    p_home = float(np.sum(margin > 0) / n)
    p_away = float(np.sum(margin < 0) / n)
    p_tie = float(np.sum(margin == 0) / n)

    ou = {"Linea": linea_ou, "Prob Over": None, "Prob Under": None, "Prob Push": None}
    if linea_ou is not None and pd.notna(linea_ou):
        line = float(linea_ou)
        ou.update({
            "Prob Over": round(float(np.sum(total > line) / n) * 100, 2),
            "Prob Under": round(float(np.sum(total < line) / n) * 100, 2),
            "Prob Push": round(float(np.sum(total == line) / n) * 100, 2),
        })

    spread = {"Linea nflverse": spread_local, "Cubre Local": None, "Cubre Visita": None, "Push": None}
    if spread_local is not None and pd.notna(spread_local):
        line = float(spread_local)
        adjusted = margin - line
        spread.update({
            "Cubre Local": round(float(np.sum(adjusted > 0) / n) * 100, 2),
            "Cubre Visita": round(float(np.sum(adjusted < 0) / n) * 100, 2),
            "Push": round(float(np.sum(adjusted == 0) / n) * 100, 2),
        })

    return {
        "Disponible": True,
        "Metodo": "Distribucion empirica de marcadores reales",
        "Muestras_Local": int(len(h_off)),
        "Muestras_Visita": int(len(a_off)),
        "Proyeccion_Score": {
            local: round(float(np.mean(score_h)), 2),
            visita: round(float(np.mean(score_a)), 2),
            "Total_Proyectado": round(float(np.mean(score_h) + np.mean(score_a)), 2),
        },
        "Moneyline": {
            "Gana Local": round(p_home * 100, 2),
            "Gana Visita": round(p_away * 100, 2),
            "Empate": round(p_tie * 100, 2),
        },
        "Over_Under": ou,
        "Spread": spread,
    }
