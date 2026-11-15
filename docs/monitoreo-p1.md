# Monitoreo de P1 con Evidently y comparación A/B

La actividad de observabilidad se implementa para P1 (regresión diaria de salidas) usando la versión de datos disponible en el manifiesto. La partición temporal existente divide del 1 de octubre de 2024 al 30 de septiembre de 2026 en 70/15/15 por días completos. Por el cálculo `int()` de esas proporciones, los intervalos inclusivos resultan:

| Uso | Periodo | Filas para el análisis |
|---|---|---|
| Referencia | 2024-10-01 a 2026-02-22 (train) | Features P1 de estaciones observadas durante el entrenamiento. |
| Reentrenamiento de candidata | train + validación, hasta 2026-06-11 | La validación se agrega al histórico para adaptar la candidata sin usar el periodo de evaluación. |
| Actual simulado / evaluación final | 2026-06-12 a 2026-09-30 (test) | Las mismas filas evalúan la versión del alias `production` y la candidata. |

El corte posterior representa un periodo de producción simulado con datos abiertos históricos; no son registros vivos de la API. Ambos modelos se evalúan con predicción diaria a un paso: cada fecha usa las observaciones reales disponibles hasta el día anterior. La población se limita a estaciones con historia en train, porque el modelo registrado no conoce estaciones nuevas. El resumen indica cuántos viajes de origen quedaron fuera de ese alcance. El reporte incluye las variables de entrada de P1 y las salidas observadas (`departures`) para analizar también drift de la variable objetivo.

Instala las dependencias de monitoreo fijadas en `uv.lock`:

```bash
uv sync --locked --extra monitoring
```

Desde el contenedor de trabajo, con el servidor de MLflow ya iniciado, define `MLFLOW_TRACKING_URI` y `MLFLOW_EXPERIMENT_NAME` según la [guía de Docker](jupyter-contenedor.md), ejecuta la [ingesta](datos-mibici.md#ingesta-de-datos) si el clon está limpio y confirma que `mibici-p1` tenga el alias `production`. Después ejecuta la entrada `monitor-p1` del MLflow Project:

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e monitor-p1 -P candidate_model=ridge -P alpha=1.0
```

También puedes probar `candidate_model=linear_regression`. La candidata Ridge se ajusta con train+validation; en ambos casos la versión vigente y la nueva se comparan sobre el mismo test, las mismas estaciones y las mismas variables. El sistema calcula RMSE, MAE y R², y calcula un intervalo de confianza del 95% para `RMSE_candidata - RMSE_producción` con 2,000 remuestreos pareados en bloques móviles de siete días (semilla 42); todas las estaciones del mismo día permanecen juntas. Si el intervalo queda debajo de cero, recomienda la candidata; si queda arriba de cero, recomienda producción; si incluye cero, reporta resultado inconcluso y recomienda conservar producción.

El comando registra la candidata como una nueva versión de `mibici-p1` con alias `ab-candidate`; conserva el alias `production` para evitar cambiar el servicio antes de revisar la evidencia. MLflow guarda en la corrida los datos y periodos usados, métricas, versiones de modelo, semilla, recomendación y artefactos. Los archivos locales se generan en `reports/monitoring/p1/`:

- `evidently_drift_quality.html`: reporte gráfico de drift por variable y calidad de datos.
- `evidently_drift_quality.json`: resultados completos de Evidently.
- `calidad_referencia.html` y `calidad_actual.html`: calidad por separado de train y del periodo posterior.
- `interpretacion_ab.md`: variables señaladas, implicaciones para P1, métricas, intervalo y recomendación.
- `predicciones_ab.csv`: predicción observada y ambas predicciones para las mismas fechas y estaciones.
- `resultado_ab.json`: resumen reproducible de periodos, versión de datos, métricas y recomendación.

### Resultado de referencia de esta ejecución

Con `data_version=d96c0ee1fac3`, Evidently comparó 184,944 filas de referencia con 41,181 filas actuales. No encontró nulos ni filas duplicadas en ninguno de los dos conjuntos. Ninguna de las seis columnas (`station_id`, `day_of_week`, `departures_lag_1`, `departures_lag_7`, `departures_mean_7`, `departures`) superó el umbral 0.10. Sus distancias respectivas fueron 0.0312, 0.0084, 0.0273, 0.0380, 0.0359 y 0.0270. Así que hay diferencias pequeñas entre periodos, pero el reporte no las clasifica como drift significativo con el umbral elegido.

En la evaluación A/B del 12 de junio al 30 de septiembre de 2026, la versión vigente 3 obtuvo RMSE 12.878, MAE 7.318 y R² 0.879. La candidata Ridge (`alpha=1.0`, versión 6, alias `ab-candidate`) obtuvo RMSE 12.820, MAE 7.319 y R² 0.880. La diferencia de RMSE fue -0.058 a favor de la candidata, pero su IC 95% por bloques fue [-0.129, 0.025] e incluye cero. La evidencia es inconclusa, por lo que se conserva `production` en la versión 3.

El modelo vigente registra como datos de entrenamiento `9ca985ed8716`; el manifiesto de la ejecución actual es `d96c0ee1fac3`. La comparación local de ambos manifiestos encontró iguales los hashes de los archivos mensuales de viajes; la diferencia fue el catálogo de estaciones actualizado, que P1 no usa para construir sus variables. Se reportan ambos identificadores para que el linaje no oculte esta diferencia. La evaluación considera 371 estaciones conocidas y excluye 95,628 viajes cuyo origen es una estación nueva; esa cobertura limitada debe tenerse presente antes de llevar una versión a producción.

### Cómo interpretar el drift

Evidently define drift por cambios estadísticos entre la referencia y el periodo actual; una señal de drift por sí sola no confirma que el modelo empeoró. En P1, cambios en `station_id` apuntan a una mezcla distinta de estaciones; cambios en `day_of_week` reflejan variaciones del patrón semanal; `departures_lag_1`, `departures_lag_7` y `departures_mean_7` describen cambios en demanda reciente y semanal; cambios en `departures` muestran desplazamiento del volumen que se predice. Contrasta esos hallazgos con el RMSE y el intervalo A/B antes de proponer actualizar el alias de producción.
