# Production entrypoint for NFL Analytics V2.
# All market validation and real-data guardrails live in app_nfl_final.py.
from app_nfl_final import *  # noqa: F401,F403

# Independent real schedule view. It does not alter models or fabricate games.
from modules.nfl_calendar_ui import render_calendario_2026

render_calendario_2026()
