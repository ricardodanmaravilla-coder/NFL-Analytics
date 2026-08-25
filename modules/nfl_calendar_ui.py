import pandas as pd
import streamlit as st
import nfl_data_py as nfl


@st.cache_data(ttl=900)
def cargar_schedule_real(season: int) -> pd.DataFrame:
    """Carga el schedule real de nflverse; no rellena partidos ni fechas."""
    try:
        return nfl.import_schedules([int(season)])
    except Exception:
        return pd.DataFrame()


def semanas_regulares(schedule: pd.DataFrame) -> list[int]:
    if schedule is None or schedule.empty or "week" not in schedule.columns:
        return []
    df = schedule.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].eq("REG")]
    weeks = pd.to_numeric(df["week"], errors="coerce").dropna().astype(int)
    return sorted(w for w in weeks.unique().tolist() if 1 <= w <= 18)


def calendario_semana(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """Devuelve únicamente juegos REG reales de la semana solicitada."""
    if schedule is None or schedule.empty:
        return pd.DataFrame()
    df = schedule.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].eq("REG")]
    wk = pd.to_numeric(df.get("week"), errors="coerce")
    df = df[wk.eq(int(week))].copy()
    if df.empty:
        return df

    out = pd.DataFrame({
        "Fecha": df.get("gameday"),
        "Hora ET": df.get("gametime"),
        "Visitante": df.get("away_team"),
        "Local": df.get("home_team"),
        "Estadio": df.get("stadium"),
    })
    if "away_rest" in df.columns:
        out["Descanso visita"] = pd.to_numeric(df["away_rest"], errors="coerce")
    if "home_rest" in df.columns:
        out["Descanso local"] = pd.to_numeric(df["home_rest"], errors="coerce")
    return out.reset_index(drop=True)


def render_calendario_2026():
    st.markdown("## 📅 Calendario NFL 2026 por semana")
    sched = cargar_schedule_real(2026)
    weeks = semanas_regulares(sched)
    if not weeks:
        st.warning("El schedule regular 2026 no está disponible en nflverse en este momento.")
        return

    week = st.selectbox("Semana del calendario 2026", weeks, index=0, key="calendar_2026_week")
    games = calendario_semana(sched, week)
    if games.empty:
        st.warning(f"No se encontraron partidos REG reales para Week {week}.")
        return

    st.caption(f"Fuente: nflverse schedule real · Week {week} · {len(games)} partidos")
    st.dataframe(games, width="stretch", hide_index=True)
