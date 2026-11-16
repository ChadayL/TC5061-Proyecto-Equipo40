# Monitoreo de P1 con Evidently y comparación A/B

## Monitorear un periodo por fechas

Trabaja dentro de `ds_workspace`, con el servidor de MLflow y la [ingesta histórica](datos-mibici.md#ingesta-de-datos) listos. El test histórico de P1 va del **12 de junio al 30 de septiembre de 2026**, ambos incluidos. El comando `monitor-p1-window` evalúa una versión ya registrada sobre el periodo que elijas.

### Elegir las fechas

`start` es el primer día evaluado y `end_exclusive` es el primer día fuera de la ventana. Usa fechas en formato `AAAA-MM-DD`.

| Periodo que quieres evaluar | `start` | `end_exclusive` | Datos necesarios |
|---|---|---|---|
| Septiembre | `2026-09-01` | `2026-10-01` | Ingesta histórica |
| Test histórico completo (111 días) | `2026-06-12` | `2026-10-01` | Ingesta histórica |
| Octubre | `2026-10-01` | `2026-11-01` | Ingesta histórica y octubre |
| Septiembre y octubre | `2026-09-01` | `2026-11-01` | Ingesta histórica y octubre |
| Test histórico más octubre | `2026-06-12` | `2026-11-01` | Ingesta histórica y octubre |

El pipeline también necesita los siete días anteriores a `start` para calcular los rezagos del primer día. Por ejemplo, para empezar el 1 de septiembre lee historial desde el 25 de agosto. Esos días se usan como entrada del modelo. Las gráficas y métricas empiezan en `start`.

### Ejecutar la ventana

Si tu ventana incluye octubre y aún falta su Parquet, ingiere el mes primero:

```bash
uv run --locked mlflow run . --env-manager local -e ingest-monitoring-month \
  --experiment-name "$MLFLOW_EXPERIMENT_NAME" -P month=2026-10
```

Para evaluar septiembre y octubre juntos con `mibici-p1` v1, ejecuta:

```bash
uv run --locked --extra monitoring mlflow run . --env-manager local -e monitor-p1-window \
  --experiment-name "$MLFLOW_EXPERIMENT_NAME" \
  -P start=2026-09-01 -P end_exclusive=2026-11-01 -P model_version=1
```

Para probar otro periodo, sustituye `start` y `end_exclusive` con los valores de la tabla. Para probar otro modelo, usa el número de una versión registrada de `mibici-p1` en **Models** como `model_version`. El inicio debe ser posterior al último día de entrenamiento de esa versión. Si falta un Parquet para la ventana o sus rezagos, la corrida indica el archivo que debes ingerir.

El pipeline pasa los viajes por `common` y construye las mismas variables diarias de P1. Cada predicción usa las salidas observadas hasta el día anterior.

Evidently compara las variables y las salidas observadas entre la ventana y la referencia reconstruida. Esta referencia usa las fechas de entrenamiento de la corrida de origen cuando están registradas, con el train histórico del proyecto como respaldo. El modelo registrado genera las predicciones y se calculan RMSE, MAE y R² sobre la ventana. Esta corrida conserva la partición 70/15/15 y el manifiesto de la ingesta histórica.

### Revisar los resultados

En MLflow, abre la corrida de tu ventana, por ejemplo `p1_monitoring_2026-09-01_v1`, y entra a **Artifacts → monitoring**. Si ejecutaste varias ventanas con el mismo inicio y versión, comprueba `end_exclusive` en **Parameters** para identificar la que quieres revisar.

- `salidas_diarias.png` compara los totales reales y predichos por día para las estaciones evaluadas. En el total, los errores de estaciones distintas se pueden compensar.
- `real_vs_predicho.png` muestra la concentración de predicciones por estación y día alrededor de la diagonal ideal, junto con la distribución de los errores.
- `predicciones.csv` contiene el valor real y el predicho por estación y día.
- `evidently_drift_quality.html` muestra calidad y drift. `resultado.json` contiene las métricas y fechas de la evaluación.

Una copia queda en `reports/monitoring/p1/windows/<run_id>/`.

## Comparación A/B con la partición histórica

La actividad de observabilidad se implementa para P1 (regresión diaria de salidas) usando la versión de datos disponible en el manifiesto. La partición temporal existente divide del 1 de octubre de 2024 al 30 de septiembre de 2026 en 70/15/15 por días completos. Por el cálculo `int()` de esas proporciones, los intervalos inclusivos resultan:

| Uso | Periodo | Filas para el análisis |
|---|---|---|
| Referencia | 2024-10-01 a 2026-02-22 (train) | Features P1 de estaciones observadas durante el entrenamiento. |
| Reentrenamiento de candidata | train + validación, hasta 2026-06-11 | La validación se agrega al histórico para adaptar la candidata sin usar el periodo de evaluación. |
| Actual simulado / evaluación final | 2026-06-12 a 2026-09-30 (test) | Las mismas filas evalúan la versión del alias `production` y la candidata. |

El corte posterior representa un periodo de producción simulado con datos abiertos históricos; no son registros vivos de la API. Ambos modelos se evalúan con predicción diaria a un paso: cada fecha usa las observaciones reales disponibles hasta el día anterior. La población se limita a estaciones con historia en train, porque el modelo registrado no conoce estaciones nuevas. El resumen indica cuántos viajes de origen quedaron fuera de ese alcance. El reporte incluye las variables de entrada de P1 y las salidas observadas (`departures`) para analizar también drift de la variable objetivo.

Desde el contenedor de trabajo, con el servidor de MLflow ya iniciado, define `MLFLOW_TRACKING_URI` y `MLFLOW_EXPERIMENT_NAME` según la [guía de Docker](jupyter-contenedor.md), ejecuta la [ingesta](datos-mibici.md#ingesta-de-datos) si el clon está limpio y confirma que `mibici-p1` tenga el alias `production`. Después ejecuta la entrada `monitor-p1` del MLflow Project:

```bash
uv run --locked --extra monitoring mlflow run . --env-manager local -e monitor-p1 \
  --experiment-name "$MLFLOW_EXPERIMENT_NAME" -P candidate_model=ridge -P alpha=1.0
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
