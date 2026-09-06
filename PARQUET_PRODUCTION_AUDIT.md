# NFL Parquet production audit

Audit date: 2026-09-06

## Findings

- The repository already had a partitioned PBP Parquet lake and DuckDB reader, but production web entrypoints still depended on the CSV path.
- Cloud Run could keep using a bundled historical snapshot indefinitely after the weekly data workflow updated the repository.

## Changes in this branch

- Prefer aggregated Parquet PBP, with CSV fallback and an explicit safe empty result when neither storage is available.
- Moneyline runtime auto-loads PBP when the web layer passes none, then filters strictly to `game_id` values already present in the allowed historical game set.
- Cloud Run loads validated fresh history from `main` with atomic local fallback and a TTL-based model/history cache.
- Weekly data workflow builds and validates the aggregated Parquet representation and persists it together with the CSV backup.
- Big-data CI verifies Parquet selection, fallback, TTL refresh wiring, temporal filtering, PBP auto-load, and production Moneyline PBP use.

## Guardrail

The runtime never admits a PBP row whose `game_id` is outside the historical game set supplied to training. This prevents storage auto-load from creating future-data leakage.
