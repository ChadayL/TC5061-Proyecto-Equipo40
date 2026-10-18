# MiBici — Proyecto TC5061

Proyecto colaborativo para estudiar los viajes de MiBici y responder tres preguntas:

- **P1 · Regresión:** predecir cuántos viajes se iniciarán por estación y día.
- **P2 · Clasificación:** aproximar el riesgo de falta de bicicletas o anclajes.
- **P3 · Clustering:** agrupar estaciones con patrones de uso similares.

El inventario real de bicicletas no está disponible en los archivos de viajes. La etiqueta de P2 es una aproximación que todavía debe justificarse.

## Instalación

Ejecutar desde la raíz del repositorio, donde está `pyproject.toml`:

```bash
python -m pip install uv
uv sync --locked
```

La versión de Python (3.10) se define en `pyproject.toml` y las dependencias en `uv.lock`.

## Ingesta de datos

Fuente: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Un solo comando, sin pasos manuales:

```bash
uv run python -m mibici.dataset ingest
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

Un solo punto de entrada para las tres preguntas:

```bash
uv run python -m mibici.modeling.train --problem p1
uv run python -m mibici.modeling.train --problem p2
uv run python -m mibici.modeling.train --problem p3
```

| Problema | Modelo por defecto | Datos |
|---|---|---|
| `p1` | Regresión lineal | Entrena con train y evalúa en validation, prediciendo a un día. |
| `p2` | Random Forest | Entrena con train y evalúa en validation; cada fila se asigna por su fecha objetivo. |
| `p3` | K-Means | Solo train. |

Cada ejecución registra parámetros, métricas y artefactos en MLflow. El servidor y el experimento se configuran con `MLFLOW_TRACKING_URI` (por defecto `http://mlflow:5000`) y `MLFLOW_EXPERIMENT_NAME`.

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
