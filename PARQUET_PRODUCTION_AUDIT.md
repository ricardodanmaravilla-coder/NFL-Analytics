# NFL Parquet production audit

Audit date: 2026-09-06

## Findings

- The repository already had a partitioned PBP Parquet lake and DuckDB reader, but production web entrypoints still depended on the CSV path.
- The current repository snapshot did not contain `data/historico_nfl_pbp_team_game.csv`, so production could silently degrade to a model without PBP.
- The weekly data workflow generated the PBP CSV but did not persist the validated aggregated Parquet dataset.

## Changes in this branch

- Prefer aggregated Parquet PBP, with CSV fallback and an explicit safe empty result when neither storage is available.
- Moneyline runtime auto-loads PBP when the web layer passes none, then filters strictly to `game_id` values already present in the allowed historical game set.
- Weekly data workflow builds and validates the aggregated Parquet representation and persists it together with the CSV backup.
- Big-data CI now verifies Parquet selection, temporal filtering, PBP auto-load, and that the production Moneyline runtime actually trains with PBP.

## Guardrail

The runtime never admits a PBP row whose `game_id` is outside the historical game set supplied to training. This prevents the storage auto-load from creating future-data leakage.
