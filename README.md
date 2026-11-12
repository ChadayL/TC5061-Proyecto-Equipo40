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

## Monitoreo de P1 con Evidently y comparación A/B

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

Inicia el servidor de MLflow, define `MLFLOW_TRACKING_URI` y `MLFLOW_EXPERIMENT_NAME` como en la sección anterior, ingiere los datos si el clon está limpio y confirma que `mibici-p1` tenga el alias `production`. Después ejecuta la entrada `monitor-p1` del MLflow Project:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e monitor-p1 -P candidate_model=ridge -P alpha=1.0
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

## Ingesta de datos

Fuente: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Un solo comando, sin pasos manuales:

```bash
uv run mlflow run --experiment-name mibici-equipo40 . --env-manager local -e ingest
```

1. **Descarga** a `data/raw/` los 24 meses de viajes (`datos_abiertos_2024_10.csv` … `datos_abiertos_2026_09.csv`) y el catálogo de estaciones `nomenclatura_2026_10.csv`. Omite los archivos que ya existen.
2. **Normaliza** cada archivo: su codificación (UTF-8 o cp1252), nombres de columna sin espacios y fechas en sus dos formatos (`dd/mm/aaaa HH:MM` en octubre de 2024, `aaaa-mm-dd HH:MM:SS` en el resto) a `datetime`.
3. **Guarda** un Parquet por mes en `data/interim/viajes/` y el catálogo en `data/interim/estaciones.parquet`.
4. **Versiona** los datos en [`references/data_manifest.json`](references/data_manifest.json), que sí se sube a Git: número de registros y sha256 de cada archivo, más una `data_version` que se registra en cada run de MLflow. Si MiBici cambia algún archivo, cambian su hash y la versión, y `git diff` del manifiesto lo muestra.

Resultado de referencia con el catálogo anterior: 24 meses, 9,043,925 viajes, 484 estaciones, versión `9ca985ed8716`. El catálogo de octubre contiene 470 estaciones. La versión efectiva se genera al ejecutar la ingesta.

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


## Pruebas, calidad de datos y seguridad

Instala el grupo de calidad (PyTest, Deepchecks, Bandit y pip-audit) con las versiones fijadas en `uv.lock`:

```bash
uv sync --locked --extra quality
```

Corre la suite automatizada desde la raíz del repositorio:

```bash
uv run pytest -q
```

Las pruebas unitarias cubren funciones de P1, P2 y P3, datos límite y la consistencia de predicciones del Random Forest con la misma semilla. Las pruebas que necesitan Parquet local se omiten en un clon limpio; para habilitarlas, primero ejecuta la ingesta indicada arriba. No subas `data/` a Git.

Ejecuta los chequeos de Deepchecks con muestras deterministas de hasta 100,000 viajes por partición:

```bash
uv run --extra quality python -m mibici.validation
```

El comando genera `reports/data_quality/integridad_train.html`, `comparacion_train_validation.html` y `resumen.json`. Deepchecks revisa integridad en train y compara train contra validation para buscar problemas de partición y drift. El resumen reporta tipos, nulos y duplicados; los viajes, identificadores de usuario y reportes generados no se versionan. Si los datos locales aún no están ingeridos, la validación indica los archivos que faltan.

En una ejecución de referencia con 1,000 filas (--sample-rows 1000), PyTest terminó con **19 pruebas aprobadas**; Deepchecks ejecutó las suites de integridad y comparación (12 chequeos cada una). En esa muestra se observaron nulos en Genero y filas duplicadas; son hallazgos reportados para análisis, no filas eliminadas automáticamente. Bandit terminó sin hallazgos de seguridad reportables; hay supresiones explícitas para el acceso al host fijo MiBici y el comando fijo git rev-parse (sin shell ni entrada del usuario).

Escanea vulnerabilidades conocidas de dependencias y patrones de seguridad en el código:

```bash
uv run --extra quality pip-audit
uv run --extra quality bandit -r mibici
```

`pip-audit` puede encontrar vulnerabilidades en versiones fijadas: revisa el paquete y el aviso antes de actualizar y vuelve a ejecutar las pruebas. Bandit puede señalar patrones que requieren inspección manual. El repositorio ignora `.env`, bases MLflow, datos y reportes locales; no pongas tokens ni contraseñas en el código. La integración de GitHub Actions ejecuta PyTest, Bandit y pip-audit en pushes y pull requests.

Para la evidencia de entrega, guarda una captura de la salida de `uv run pytest -q`, de los dos escaneos de seguridad y del reporte HTML de Deepchecks abierto en el navegador. En GitHub, la pestaña **Actions** muestra el resultado automatizado del workflow `quality`.
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

- [Servir el modelo P1 con FastAPI y Docker Compose](docs/servir-p1.md).

- [Cómo funcionan los stages y cómo integrar transformaciones](docs/flujo_datos.md).
- [Cómo registrar hallazgos](references/hallazgos/README.md).

Trabajar en ramas por tarea (`feat/`, `fix/`, `docs/`) e integrar mediante PR revisado. Coordinar una persona a la vez por notebook y usar nombres `1.0-iniciales-descripcion.ipynb`.

Excluir del historial de Git los datos, modelos serializados, `.env`, `mlruns/`, `mlflow.db` y `mlartifacts/`. Verificar el `.gitignore` antes de agregar archivos.

## Estado

- **Implementado:** ingesta reproducible (descarga, Parquet y manifiesto versionado), lectura por partición, stages `structure` y `quality_fixes`, P1 completo, P2 y P3 integrados en el CLI con registro en MLflow.
- **Pendiente en P2:** fórmula y umbral de riesgo, y la inclusión de estaciones sin actividad.
- **Pendiente en P3:** normalizar por días activos y definir el uso del catálogo de estaciones.
- **Otros:** límites de edad (H-co-006), elegibilidad de estaciones en el calendario de P1 (H-co-010).

## Despliegue del modelo P1 con FastAPI y Docker

El proyecto expone el modelo de regresión P1 mediante una API REST desarrollada con FastAPI, validación Pydantic y un servicio de inferencia MLflow.

### Estrategia de despliegue

Se utiliza **inferencia en tiempo real**, ya que cada petición HTTP recibe una predicción inmediata de salidas de bicicletas para una estación y sus variables de entrada.

El modelo se recupera desde **MLflow Model Registry** por nombre y versión (`models:/mibici-p1/1`) y se empaqueta dentro de la imagen Docker de inferencia. Por lo tanto, los contenedores no necesitan conectarse al Registry durante cada predicción.

### 1. Descargar el modelo registrado

Con el servidor MLflow disponible y el modelo `mibici-p1` registrado:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

uv run python -c '
from pathlib import Path
import mlflow

destination = Path("build/serving")
destination.mkdir(parents=True, exist_ok=True)

mlflow.artifacts.download_artifacts(
    artifact_uri="models:/mibici-p1/1",
    dst_path=str(destination)
)
'
```

Verificar que existan `build/serving/MLmodel` y `build/serving/requirements.txt`.

### 2. Construir las imágenes OCI

Se utiliza el archivo `Dockerfile`, compatible con Docker, y `docker-compose.serving.yml`.

```bash
docker compose -f docker-compose.serving.yml build
```

Se construyen las imágenes:

- `mibici-api:local`: aplicación FastAPI.
- `mibici-inference:p1-v1`: modelo P1 servido mediante MLflow.

### 3. Levantar los contenedores

```bash
docker compose -f docker-compose.serving.yml up -d
docker compose -f docker-compose.serving.yml ps
```

Servicios publicados en el equipo local:

- FastAPI: `http://127.0.0.1:8000`
- MLflow Inference: `http://127.0.0.1:8080`

Documentación interactiva de la API: `http://127.0.0.1:8000/docs`.

### 4. Consumir la API

Ejemplo de petición HTTP:

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": 2,
    "day_of_week": 1,
    "departures_lag_1": 76.0,
    "departures_lag_7": 21.0,
    "departures_mean_7": 55.857142857142854
  }'
```

Respuesta obtenida durante la validación:

**HTTP 200 OK**

```json
{
  "predicted_departures": 46.298935294225515
}
```

FastAPI valida las entradas con Pydantic y devuelve la predicción en JSON. Los datos inválidos generan respuestas de validación HTTP 422.

### 5. Evidencia y pruebas

- [Captura de la API respondiendo desde Docker](evidence/p1-docker-api-success.png).
- [Respuesta JSON de inferencia real](evidence/p1-docker-prediction.json).
- [Reporte técnico de validación](docs/validacion-despliegue-p1.md).
- [Documentación adicional del servicio](docs/servir-p1.md).

Durante la validación local se ejecutaron 34 pruebas automatizadas satisfactoriamente y se comprobó una petición real HTTP 200 contra el modelo P1 ejecutándose en Docker.

### 6. Detener los servicios

```bash
docker compose -f docker-compose.serving.yml down
```

