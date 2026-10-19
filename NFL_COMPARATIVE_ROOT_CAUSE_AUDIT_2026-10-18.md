# Auditoría comparativa NFL — 2026-10-18

## Estado verificado
- NFL Independiente: 31 recomendaciones con resultado liquidado, 16 WIN, 15 LOSS; ROI unitario -4.93%.
- ML: 9, 5 WIN, ROI -7.52%; SPREAD: 14, 8 WIN, ROI +10.92%; TOTAL: 8, 3 WIN, ROI -29.74%.
- Las 31 recomendaciones anunciaban 19.63 victorias esperadas, frente a 16 observadas. No inferir significancia con esta muestra.
- La inversión de todas las apuestas no es una estrategia validada; calcularla con las cuotas del lado opuesto, no negando el ROI original.

## Hallazgos en código
1. `nfl_independent_live_monitor.py`: Moneyline utiliza `predict_proba` crudo, sin calibración OOS; por tanto `edge_pp` y Kelly se apoyan en probabilidades no verificadas.
2. `history_and_future`: actualiza marcadores del año corriente desde schedules, pero el propio campo de provenance declara `current_season_qb_pbp_verified=False`. La forma reciente de QB y PBP puede quedar desfasada.
3. `offer`: umbrales 58% / 3 pp / 3% EV sobre estimaciones puntuales; no hay penalización explícita por error de calibración o incertidumbre.
4. `backtest_nfl_independent_walkforward.py`: cuotas históricas aproximadas, sin prueba de disponibilidad al momento de apostar; 2025 es un único holdout. Su ROI no equivale a rentabilidad ejecutable.
5. `audit_nfl_independent_probabilities.py`: reconoce `odds_snapshot_verified=False`, `weather_forecast_timestamp_verified=False` y `qb_starter_confirmed_pregame=False`. Usa umbral 55% y ML de cuota negativa para el subconjunto `production_like`, distinto del independiente live (58% y sin restricción de cuota negativa). Comparar sus ROIs directamente sería inválido.
6. `backtest_nfl_production_all_markets.py`: sistema principal con `MIN_P=58`, `MIN_MC=58`, `MAX_DISAGREE=15`; además añade variantes calibradas de Spread/Total. No es la misma política de selección del Independiente; requiere evaluación paralela con cuotas y periodos idénticos.
7. La evaluación de picks del mismo partido/mercados correlacionados no debe tratar cada pick como observación independiente al calcular incertidumbre.

## Decisiones
- No declarar ventaja en ML, SPREAD ni TOTAL por un resultado de seis semanas.
- Congelar recomendaciones ya emitidas; no alterar su probabilidad, cuota, línea o ID.
- Priorizar verificación temporal de cuotas y datos, calibración OOS de ML y pruebas de estabilidad por temporada.
- Para comparación justa: ambos sistemas sobre el mismo slate y timestamps, ROI a unidad fija, Brier/log-loss, calibración por bins, closing-line value y bootstrap agrupado por partido/semana.
- Estudiar estrategia invertida con precios del lado contrario y fuera de muestra; no recomendar invertir picks por intuición.
- Hasta entonces mantener el Independiente en PAPER_ONLY. No aumentar stakes ni forzar picks.

## Alcance
Esta es una auditoría estática del código y de los resultados liquidados disponibles; **no** es un backtest nuevo ni una verificación de rentabilidad del sistema principal. No se modificaron filtros ni recomendaciones existentes.
