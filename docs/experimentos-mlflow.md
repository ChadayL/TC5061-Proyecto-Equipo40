# Entrenamiento y experimentos con MLflow

Sigue la [guía de Docker](jupyter-contenedor.md) para iniciar Jupyter, el servidor de tracking y su almacenamiento. El entorno usa Python 3.10 y las dependencias de `uv.lock`. Comprueba que `MLFLOW_EXPERIMENT_NAME` apunta al experimento donde quieres registrar las corridas.

Para revisar las fuentes, las particiones o los formatos de los archivos, consulta la [guía de datos](datos-mibici.md).

## Proyecto MLflow reproducible

`MLproject` ofrece los puntos de entrada `ingest` y `train`. Ejecuta lo siguiente desde la raíz del repositorio y dentro del contenedor de trabajo:

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e ingest
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p1
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p2
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p3
```

Se pueden variar modelos e hiperparámetros desde el mismo punto de entrada; los valores omitidos se toman de `mibici/config.py`:

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p1 -P fit_intercept=false
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p2 -P n_estimators=300 -P max_depth=20
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p3 -P n_clusters=4
```

Los modelos disponibles son `linear_regression` y `ridge` para P1, `random_forest` para P2 y `kmeans` para P3. P3 elige automáticamente K por silhouette si no se especifica `n_clusters`. Todas las corridas registran commit, equipo, parámetros, métricas y versión de datos en el experimento configurado. La ingesta debe ejecutarse antes del entrenamiento en un clon limpio.

### Modelos por defecto

| Problema | Modelo por defecto | Datos |
|---|---|---|
| `p1` | Regresión lineal | Train y validation; predicción diaria a un paso. |
| `p2` | Random Forest | Train y validation; cada fila se asigna por su fecha objetivo. |
| `p3` | K-Means | Solo train; selección de K por silhouette salvo que se fije `n_clusters`. |

### Experimentos con corridas anidadas

Los planes en `configs/experiments/` contienen tres configuraciones candidatas por pregunta. Los parámetros omitidos mantienen los valores de `config.py`. Se ejecutan secuencialmente para no competir por memoria:

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e search -P config=configs/experiments/p1.yaml
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e search -P config=configs/experiments/p2.yaml
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e search -P config=configs/experiments/p3.yaml
```

La búsqueda es la corrida padre (`stage_of_work=search`). Cada combinación ejecuta el punto de entrada `train` del mismo Project y aparece como corrida hija (`stage_of_work=candidate`). No se cambian los cortes temporales ni las métricas anteriores. P1 compara Ridge con distintos valores de `alpha`; P2 varía árboles y profundidad; P3 fija K. Los YAML son planes iniciales, no modelos ganadores ni resultados ya obtenidos.

Cada candidata conserva el modelo con firma y ejemplo, `plots/` (desempeño), `data/` (manifiesto y resultados agregados) y `config/` (valores efectivos, configuración y dependencias). También registra metadatos del conjunto de entrenamiento con `log_input`, sin subir la tabla de viajes, y etiquetas `question`, `model_family`, `stage_of_work`, `team` y `commit`. El hash y la copia de `features.py` permiten detectar cambios en variables entre corridas.

Para entrenar una candidata individual o reproducir una configuración base:

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p1 -P model=ridge -P alpha=1.0
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e train -P question=p1 -P model=linear_regression -P stage_of_work=baseline
```

La etiqueta `baseline` solo identifica la configuración ejecutada; no reemplaza ni modifica la corrida de la entrega anterior. Antes de comparar, comprueben la misma versión de datos, variables, población de estaciones y particiones. Si cambia la preparación, repitan la configuración base y documenten el cambio. Las métricas de P1/P2 siguen siendo de validation y silhouette de P3 sigue siendo de train. La selección y la evaluación final de test se realizan después de cerrar las configuraciones; estas búsquedas no consumen test ni promueven modelos automáticamente.

El tiempo `fit_seconds` mide el ajuste del modelo; en P3 incluye escalado y exploración de K. `predict_seconds` mide las predicciones de validación de P1/P2 (incluye probabilidades en P2). Son mediciones de una ejecución, no una prueba de estabilidad entre semillas.

### Configuración seleccionada para P1

La mejor corrida de P1 en la búsqueda histórica de octubre fue **Ridge con `alpha=10.0` y `fit_intercept=true`**. Se eligió por el menor RMSE de validación entre la regresión lineal de referencia y las tres configuraciones de Ridge.

| Dato | Valor de referencia |
|---|---|
| RMSE de validación | 13.1210 (referencia lineal: 13.1798) |
| MAE de validación | 7.0709 (referencia lineal: 7.1096) |
| Experimento histórico | `TC5061-Proyecto-Equipo40` |
| Run ID de la corrida seleccionada | `8b6a9848545c45d8b23b234fe63914cb` |
| Commit registrado | `7c4f2bc91a955279da371d8a533179e6521d62a0` |
| Versión del manifiesto histórico | `9ca985ed8716` |
| Entrenamiento | 2024-10-01 a 2026-02-22 |
| Validación | 2026-02-23 a 2026-06-11 |

Con los datos ya ingeridos, ejecuta desde la raíz del repositorio **dentro del contenedor de trabajo**. Define `MLFLOW_EXPERIMENT_NAME` con el nombre del experimento de tu servidor.

```bash
uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" \
  --env-manager local -e train -P question=p1 -P model=ridge \
  -P alpha=10.0 -P fit_intercept=true -P stage_of_work=candidate
```

El comando repite la configuración elegida directamente. Para reproducir la corrida histórica, usa también el commit y la versión del manifiesto de la tabla. Si trabajas con el manifiesto actual, registra su versión efectiva y verifica las métricas de la nueva corrida.

Para comparar resultados entre integrantes, comparte el Run ID y la versión de datos de la corrida usada. Si todos necesitan puntuar con el mismo artefacto, usen un Registry común y la referencia fija `models:/mibici-p1/<versión>`. Cada Registry local tiene su propio alias `production`. El [A/B de monitoreo](monitoreo-p1.md#resultado-de-referencia-de-esta-ejecución) corresponde a una ejecución posterior con una vigente v3 y una candidata Ridge con `alpha=1.0`.

### Registrar los modelos seleccionados

Después de entrenar y comparar las candidatas, registra el modelo elegido de cada pregunta. Ejecuta estos comandos dentro del contenedor de trabajo y usa el mismo experimento que contiene los baselines. El ejemplo usa el nombre del experimento histórico. Si tu experimento tiene otro nombre, usa ese valor.

```bash
export MLFLOW_TRACKING_URI=http://mlflow:5000
export MLFLOW_EXPERIMENT_NAME=TC5061-Proyecto-Equipo40

uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p1
uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p2
uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p3
```

La selección minimiza `validation_rmse` en P1 y maximiza `f1` en P2 y `silhouette_score` en P3. Cada versión conserva la corrida de origen y etiquetas con la versión de datos y el commit. La promoción asigna el alias `production` en el Registry. En MLflow 3, el alias es la referencia recomendada para cargar el modelo.

### Ciclo de vida y carga por alias

Usamos `production` como alias principal porque separa el nombre que utiliza quien consume el modelo de su número de versión. Al elegir otro modelo, se mueve el alias a la nueva versión; las anteriores conservan su linaje y pueden recuperarse. Las primeras versiones registradas también conservaron la etapa clásica `Production`. Para resolver el modelo vigente se usa el alias. Registrar o promover un modelo no levanta un servicio de inferencia.

| Pregunta | Modelo registrado | Referencia de carga |
|---|---|---|
| P1 | `mibici-p1` | `models:/mibici-p1@production` |
| P2 | `mibici-p2` | `models:/mibici-p2@production` |
| P3 | `mibici-p3` | `models:/mibici-p3@production` |

Desde Python dentro del contenedor, con el entorno del proyecto:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
model = mlflow.pyfunc.load_model("models:/mibici-p1@production")
predictions = model.predict(features)
```

`features` debe ser una tabla con las columnas de la firma del modelo: cinco variables en P1, once en P2 y ocho en P3. P1 y P3 incluyen sus transformaciones de encoding/escalado o escalado, respectivamente. Las agregaciones desde viajes crudos se preparan antes de llamar al modelo. No se necesita la columna objetivo para predecir.

Se verificó dentro de Docker la carga de las versiones 1 de los tres modelos por alias `production`, con sus firmas y ejemplos de entrada guardados, y la generación de predicciones. Esta comprobación valida la carga e inferencia técnica; no sustituye la evaluación final en test ni el análisis de estabilidad.

Cada ejecución de `promote` crea una versión, incluso si vuelve a seleccionar la misma corrida. No lo repitas solo para comprobar la carga: usa `load_model` con el alias.
