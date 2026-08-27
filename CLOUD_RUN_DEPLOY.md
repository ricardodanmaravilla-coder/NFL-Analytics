# NFL Analytics en Google Cloud Run

## Arquitectura
- Frontend HTML ligero servido por FastAPI.
- Endpoint `GET /api/scan/{season}/{week}`.
- MoneylineRuntime conserva el modelo validado de margen/PBP.
- Modelos cacheados por temporada/semana dentro de la instancia.
- Un worker Uvicorn para limitar RAM.

## Despliegue recomendado

1. Crea o selecciona un proyecto en Google Cloud y activa facturación.
2. Abre Cloud Shell.
3. Clona el repositorio y entra en él.
4. Ejecuta:

```bash
gcloud run deploy nfl-analytics \
  --source . \
  --region northamerica-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 1 \
  --timeout 900 \
  --max-instances 2 \
  --set-env-vars OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1
```

Para la primera prueba, `4Gi`, `2 CPU` y `concurrency=1` priorizan estabilidad. Después de medir memoria real podemos reducir recursos.

## Verificación
- `/health` debe responder `status: ok`.
- `/` abre la interfaz web.
- `/api/scan/2026/1` ejecuta un escaneo.

## Siguiente optimización
El entrenamiento sigue cacheado por corte temporal. Para producción definitiva conviene generar y guardar el modelo tras cada actualización de datos, de modo que el endpoint haga sólo inferencia.
