import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ESTADIOS = {
    "BUF": (42.773, -78.786, False), "MIA": (25.957, -80.238, False), "NE": (42.090, -71.264, False),
    "NYJ": (40.813, -74.074, False), "NYG": (40.813, -74.074, False), "BAL": (39.277, -76.622, False),
    "CIN": (39.095, -84.516, False), "CLE": (41.506, -81.699, False), "PIT": (40.446, -80.015, False),
    "HOU": (29.684, -95.410, True), "IND": (39.760, -86.163, True), "JAX": (30.323, -81.637, False),
    "TEN": (36.166, -86.771, False), "DEN": (39.743, -105.020, False), "KC": (39.048, -94.483, False),
    "LV": (36.090, -115.183, True), "LAC": (33.953, -118.339, True), "DAL": (32.747, -97.092, True),
    "PHI": (39.900, -75.167, False), "WAS": (38.907, -76.864, False), "CHI": (41.862, -87.616, False),
    "DET": (42.340, -83.045, True), "GB": (44.501, -88.062, False), "MIN": (44.973, -93.257, True),
    "ATL": (33.755, -84.400, True), "CAR": (35.225, -80.852, False), "NO": (29.951, -90.081, True),
    "TB": (27.975, -82.503, False), "ARI": (33.527, -112.262, True), "LA": (33.953, -118.339, True),
    "LAR": (33.953, -118.339, True), "SF": (37.403, -121.969, False), "SEA": (47.595, -122.331, False),
}


def _num(v):
    try:
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def roof_is_dome(roof, fallback=False):
    text = str(roof or "").strip().lower()
    if text in {"dome", "closed", "indoors", "indoor"}:
        return True
    if text in {"outdoors", "outdoor", "open"}:
        return False
    return bool(fallback)


def forecast_kickoff(team, gameday, gametime, roof=None, timeout=6):
    """Return temperature °F, wind mph, dome flag, and a diagnostic message.

    nflverse `gametime` is treated as Eastern time, matching the schedule UI/runtime.
    Outdoor forecast failures are represented as missing weather rather than fabricated
    values so the model's `temp_missing`/`wind_missing` features can handle them.
    """
    info = ESTADIOS.get(team)
    if not info:
        return None, None, False, "Sin coordenadas verificadas"

    lat, lon, default_dome = info
    dome = roof_is_dome(roof, default_dome)
    if dome:
        return None, None, True, "Domo/techo cerrado"

    try:
        date = pd.to_datetime(gameday).date()
        hhmm = str(gametime or "13:00")[:5]
        et = dt.datetime.combine(date, dt.time.fromisoformat(hhmm), tzinfo=ZoneInfo("America/New_York"))
        utc = et.astimezone(ZoneInfo("UTC"))
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "start_date": utc.date().isoformat(),
            "end_date": utc.date().isoformat(),
        }
        data = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=timeout).json()
        times = pd.to_datetime(data.get("hourly", {}).get("time", []), utc=True)
        if not len(times):
            return None, None, False, "Forecast no disponible"
        idx = int(np.argmin(np.abs(times - pd.Timestamp(utc))))
        temp = _num(data["hourly"]["temperature_2m"][idx])
        wind = _num(data["hourly"]["wind_speed_10m"][idx])
        if temp is None or wind is None:
            return None, None, False, "Forecast incompleto"
        return temp, wind, False, f"{temp:.1f}°F · {wind:.1f} mph"
    except Exception:
        return None, None, False, "Forecast no disponible"
