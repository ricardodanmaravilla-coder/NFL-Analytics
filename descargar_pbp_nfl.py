import os
import pandas as pd
import nfl_data_py as nfl

from modules.nfl_pbp_engine import agregar_pbp_por_equipo_partido


def descargar_pbp_real():
    os.makedirs("data", exist_ok=True)
    temporadas = [2021, 2022, 2023, 2024, 2025, 2026]
    agregados = []

    print("🏈 Descargando play-by-play real de nflverse...")
    for season in temporadas:
        try:
            print(f"   ⏳ PBP {season}...")
            pbp = nfl.import_pbp_data([season], downcast=True, cache=False, alt_path=None)
            if pbp is None or pbp.empty:
                print(f"   ⚠️ {season}: sin datos")
                continue
            agg = agregar_pbp_por_equipo_partido(pbp)
            if agg.empty:
                print(f"   ⚠️ {season}: PBP descargado pero sin jugadas EPA utilizables")
                continue
            agregados.append(agg)
            print(f"   ✅ {season}: {len(pbp):,} jugadas -> {len(agg):,} filas equipo-partido")
        except Exception as e:
            print(f"   ⚠️ {season} no disponible: {e}")

    if not agregados:
        raise RuntimeError("No se pudo descargar ningún play-by-play real de nflverse")

    out = pd.concat(agregados, ignore_index=True)
    out = out.drop_duplicates(subset=["game_id", "team"], keep="last")
    out = out.sort_values(["season", "week", "game_id", "team"])
    path = "data/historico_nfl_pbp_team_game.csv"
    out.to_csv(path, index=False)
    print(f"✅ PBP agregado guardado: {path} ({len(out):,} filas)")


if __name__ == "__main__":
    descargar_pbp_real()
