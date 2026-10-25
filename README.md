# MiBici — Proyecto TC5061

Proyecto colaborativo para estudiar los viajes de MiBici y responder tres preguntas:

- **P1 · Regresión:** predecir cuántos viajes se iniciarán por estación y día.
- **P2 · Clasificación:** aproximar el riesgo de falta de bicicletas o anclajes.
- **P3 · Clustering:** agrupar estaciones con patrones de uso similares.

El inventario real de bicicletas no está disponible en los archivos de viajes. La etiqueta de P2 es una aproximación que todavía debe justificarse.

## Instalación y servidor MLflow

Se requiere Python 3.10. El entorno de `MLproject` fija las versiones directas usadas por el proyecto en `requirements-mlproject.txt`; `uv.lock` conserva el lock completo para el entorno de desarrollo.

```bash
python -m pip install uv
uv sync --locked
```

En una terminal, inicia el servidor local con backend SQLite y artefactos locales:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

En otra terminal, apunta el cliente al mismo servidor (PowerShell: `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"`; Bash: `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`). Por defecto se usa el experimento `mibici-equipo40`; puede cambiarse con `MLFLOW_EXPERIMENT_NAME`. La base `mlflow.db` y `mlartifacts/` son locales y están excluidos de Git.

## Proyecto MLflow reproducible

`MLproject` ofrece los puntos de entrada `ingest` y `train`. Desde la raíz, luego de iniciar el servidor, ejecutar:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e ingest
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p1
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p2
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p3
```

Se pueden variar modelos e hiperparámetros desde el mismo punto de entrada; los valores omitidos se toman de `mibici/config.py`:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p1 -P fit_intercept=false
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p2 -P n_estimators=300 -P max_depth=20
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p3 -P n_clusters=4
```

Los modelos disponibles son `linear_regression` y `ridge` para P1, `random_forest` para P2 y `kmeans` para P3. P3 elige automáticamente K por silhouette si no se especifica `n_clusters`. Todas las corridas registran commit, equipo, parámetros, métricas y versión de datos en el experimento configurado. La ingesta debe ejecutarse antes del entrenamiento en un clon limpio.

### Experimentos con corridas anidadas

Los planes en `configs/experiments/` contienen tres configuraciones candidatas por pregunta. Los parámetros omitidos mantienen los valores de `config.py`. Se ejecutan secuencialmente para no competir por memoria:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e search -P config=configs/experiments/p1.yaml
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e search -P config=configs/experiments/p2.yaml
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e search -P config=configs/experiments/p3.yaml
```

La búsqueda es la corrida padre (`stage_of_work=search`). Cada combinación ejecuta el punto de entrada `train` del mismo Project y aparece como corrida hija (`stage_of_work=candidate`). No se cambian los cortes temporales ni las métricas anteriores. P1 compara Ridge con distintos valores de `alpha`; P2 varía árboles y profundidad; P3 fija K. Los YAML son planes iniciales, no modelos ganadores ni resultados ya obtenidos.

Cada candidata conserva el modelo con firma y ejemplo, `plots/` (desempeño), `data/` (manifiesto y resultados agregados) y `config/` (valores efectivos, configuración y dependencias). También registra metadatos del conjunto de entrenamiento con `log_input`, sin subir la tabla de viajes, y etiquetas `question`, `model_family`, `stage_of_work`, `team` y `commit`. El hash y la copia de `features.py` permiten detectar cambios en variables entre corridas.

Para entrenar una candidata individual o reproducir una configuración base:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p1 -P model=ridge -P alpha=1.0
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e train -P question=p1 -P model=linear_regression -P stage_of_work=baseline
```

La etiqueta `baseline` solo identifica la configuración ejecutada; no reemplaza ni modifica la corrida de la entrega anterior. Antes de comparar, comprueben la misma versión de datos, variables, población de estaciones y particiones. Si cambia la preparación, repitan la configuración base y documenten el cambio. Las métricas de P1/P2 siguen siendo de validation y silhouette de P3 sigue siendo de train. La selección y la evaluación final de test se realizan después de cerrar las configuraciones; estas búsquedas no consumen test ni promueven modelos automáticamente.

El tiempo `fit_seconds` mide el ajuste del modelo; en P3 incluye escalado y exploración de K. `predict_seconds` mide las predicciones de validación de P1/P2 (incluye probabilidades en P2). Son mediciones de una ejecución, no una prueba de estabilidad entre semillas.

### Registrar los modelos seleccionados

Después de entrenar y comparar las candidatas, registra el modelo elegido de cada pregunta. Ejecuta estos comandos dentro del contenedor de trabajo y usa el mismo experimento que contiene los baselines. El servidor actual conserva el historial en `TC5061-Proyecto-Equipo40`:

```bash
export MLFLOW_TRACKING_URI=http://mlflow:5000
export MLFLOW_EXPERIMENT_NAME=TC5061-Proyecto-Equipo40

uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p1
uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p2
uv run --locked mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e promote -P question=p3
```

La selección minimiza `validation_rmse` en P1 y maximiza `f1` en P2 y `silhouette_score` en P3. Cada versión conserva la corrida de origen y etiquetas con la versión de datos y el commit. Se asigna el alias `production` y la etapa `Production` del Registry. En MLflow 3, el alias es la referencia recomendada para cargar el modelo.

### Ciclo de vida y carga por alias

Usamos `production` como alias principal porque separa el nombre que utiliza quien consume el modelo de su número de versión. Al elegir otro modelo, se mueve el alias a la nueva versión; las anteriores conservan su linaje y pueden recuperarse. También se mantiene la etapa clásica `Production` para compatibilidad, pero los consumidores deben cargar por alias. Registrar o promover un modelo no levanta un servicio de inferencia.

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

## Ingesta de datos

Fuente: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Un solo comando, sin pasos manuales:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e ingest
```

1. **Descarga** a `data/raw/` los 24 meses de viajes (`datos_abiertos_2024_10.csv` … `datos_abiertos_2026_09.csv`) y la nomenclatura `nomenclatura_2026_09.csv`. Omite los archivos que ya existen.
2. **Normaliza** cada archivo: su codificación (UTF-8 o cp1252), nombres de columna sin espacios y fechas en sus dos formatos (`dd/mm/aaaa HH:MM` en octubre de 2024, `aaaa-mm-dd HH:MM:SS` en el resto) a `datetime`.
3. **Guarda** un Parquet por mes en `data/interim/viajes/` y el catálogo en `data/interim/estaciones.parquet`.
4. **Versiona** los datos en [`references/data_manifest.json`](references/data_manifest.json), que sí se sube a Git: número de registros y sha256 de cada archivo, más una `data_version` que se registra en cada run de MLflow. Si MiBici cambia algún archivo, cambian su hash y la versión, y `git diff` del manifiesto lo muestra.

Resultado actual: 24 meses, 9,043,925 viajes, 484 estaciones, versión `9ca985ed8716`.

El periodo (`DATA_START`, `DATA_END`), las URLs (`SOURCE_FOLDERS`), la codificación y el formato de fecha de cada archivo se configuran en `mibici/config.py`.

### Particiones

`get_split_periods()` divide el periodo configurado por días completos (70/15/15):

| Partición | Periodo |
|---|---|
| train | 2024-10-01 a 2026-02-22 |
| validation | 2026-02-23 a 2026-06-11 |
| test | 2026-06-12 a 2026-09-30 |

Los viajes se asignan por `Inicio_del_viaje`. La carga usa `train` por defecto. Test queda reservado: no se usa para explorar ni para decidir la preparación.

## Comprobar la lectura

```bash
uv run python -m mibici.dataset check
```

Carga una muestra de cinco viajes de train, el catálogo completo (484 estaciones) y muestra la versión de los datos. Opciones: `--split` y `--nrows`.

## Trabajar en notebooks

```bash
uv run python -m ipykernel install --user --name mibici --display-name "Python (MiBici)"
```

Abrir el notebook y seleccionar **Python (MiBici)**. Los notebooks cargan los datos desde el pipeline:

```python
from mibici.pipeline import load_stage

datos = load_stage("structure")      # train por defecto
viajes = datos["viajes"]
estaciones = datos["estaciones"]
```

| Stage | Ejecución |
|---|---|
| `raw` | Archivos originales de la partición. |
| `structure` | Tipos: año de nacimiento, género, fechas y nombre de estación. |
| `quality_fixes` | Elimina duplicados y convierte años fuera del rango de edad a faltantes. |
| `common` | Todas las etapas compartidas. |

`nrows` limita el total de viajes cargados (útil para probar código) y `split` elige la partición.

## Entrenar

Usa `mlflow run` para todas las corridas; consulta la sección [Proyecto MLflow reproducible](#proyecto-mlflow-reproducible) para reproducir ingesta, entrenamientos y variantes de parámetros.

| Problema | Modelo por defecto | Datos |
|---|---|---|
| `p1` | Regresión lineal | Train y validation; predicción diaria a un paso. |
| `p2` | Random Forest | Train y validation; cada fila se asigna por su fecha objetivo. |
| `p3` | K-Means | Solo train; selección de K por silhouette salvo que se fije `n_clusters`. |

## Estructura

| Ruta | Responsabilidad |
|---|---|
| `mibici/config.py` | Rutas, periodo, particiones, archivos y MLflow. |
| `mibici/dataset.py` | Ingesta (descarga, Parquet y manifiesto), particiones temporales y carga. |
| `mibici/pipeline.py` | Orden de stages y `load_stage()`. |
| `mibici/data_preparation/` | Funciones de preparación, un `run()` por etapa, encoder y scaler. |
| `mibici/features.py` | Construcción de variables por pregunta. |
| `mibici/modeling/p1.py`, `p2.py`, `p3.py` | Datos, modelo y evaluación de cada pregunta. |
| `mibici/modeling/train.py` | Línea de comandos de entrenamiento. |
| `mibici/modeling/tracking.py` | Registro en MLflow. |
| `mibici/plots.py` | Gráficas generadas por los entrenamientos. |
| `notebooks/` | Comprensión, preparación y experimentación. |
| `references/hallazgos/` | Hallazgos de datos y su seguimiento. |
| `data/` | Datos locales (no se versionan). |

## Documentación del equipo

- [Cómo funcionan los stages y cómo integrar transformaciones](docs/flujo_datos.md).
- [Cómo registrar hallazgos](references/hallazgos/README.md).

Trabajar en ramas por tarea (`feat/`, `fix/`, `docs/`) e integrar mediante PR revisado. Coordinar una persona a la vez por notebook y usar nombres `1.0-iniciales-descripcion.ipynb`.

Excluir del historial de Git los datos, modelos serializados, `.env`, `mlruns/`, `mlflow.db` y `mlartifacts/`. Verificar el `.gitignore` antes de agregar archivos.

## Estado

- **Implementado:** ingesta reproducible (descarga, Parquet y manifiesto versionado), lectura por partición, stages `structure` y `quality_fixes`, P1 completo, P2 y P3 integrados en el CLI con registro en MLflow.
- **Pendiente en P2:** fórmula y umbral de riesgo, y la inclusión de estaciones sin actividad.
- **Pendiente en P3:** normalizar por días activos y definir el uso del catálogo de estaciones.
- **Otros:** límites de edad (H-co-006), elegibilidad de estaciones en el calendario de P1 (H-co-010).
