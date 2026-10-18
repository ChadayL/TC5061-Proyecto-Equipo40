# MiBici — Proyecto TC5061

Proyecto colaborativo para estudiar los viajes de MiBici y responder tres preguntas:

- **P1 · Regresión:** predecir cuántos viajes se iniciarán por estación o zona.
- **P2 · Clasificación:** aproximar el riesgo de falta de bicicletas o anclajes.
- **P3 · Clustering:** agrupar estaciones con patrones de uso similares.

El inventario real de bicicletas no está disponible en los archivos de viajes. La etiqueta de P2 requiere una aproximación justificada.

## Instalación

Ejecutar desde la raíz del repositorio, donde está `pyproject.toml`:

```bash
python -m pip install uv
uv sync --locked
```

Se requiere un `uv.lock` incorporado al repositorio. Si aún no existe, quien configura inicialmente el proyecto ejecuta `uv sync` y agrega el archivo mediante PR. La versión de Python se define en `pyproject.toml` y, si existe, `.python-version`.

## Preparar los datos

Fuente: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

Para la exploración inicial, colocar estos archivos sin modificar dentro de `data/raw/`:

- `datos_abiertos_2026_08.csv`
- `datos_abiertos_2026_09.csv`
- `nomenclatura_2026_09.csv`

Los nombres se configuran en `mibici/config.py`. Estos dos meses son la muestra de desarrollo; el periodo final queda por justificar.

## Ejecutar la comprobación de lectura

```bash
uv run python -m mibici.dataset
```

El comando actual carga cinco viajes por archivo y el catálogo completo. En la prueba inicial mostró 10 viajes y 484 filas del catálogo. No descarga ni transforma todo el conjunto.

## Trabajar en notebooks

Con `ipykernel` incluido en las dependencias de desarrollo:

```bash
uv run python -m ipykernel install --user --name mibici --display-name "Python (MiBici)"
```

Abrir el notebook en el editor o Jupyter disponible y seleccionar **Python (MiBici)**. Comenzar por estructura dentro de `notebooks/01_data_understanding/`.

```python
from mibici.pipeline import load_stage

datos = load_stage("raw", nrows=5)
viajes = datos["viajes"]
estaciones = datos["estaciones"]
```

Para cargar todos los registros seleccionados, omitir `nrows`. Para obtener toda la preparación compartida implementada:

```python
datos = load_stage("common")
```

## Estructura

| Ruta | Responsabilidad |
|---|---|
| `mibici/config.py` | Rutas y configuración compartida. |
| `mibici/dataset.py` | Lectura; futura ingesta y versionado. |
| `mibici/pipeline.py` | Orden de stages y `load_stage()`. |
| `mibici/data_preparation/` | Funciones de preparación y `run()` por etapa. |
| `mibici/features.py` | Construcción de variables. |
| `mibici/modeling/` | Código de entrenamiento y predicción. |
| `notebooks/` | Comprensión, preparación y experimentación. |
| `data/` | Datos locales originales, intermedios y procesados. |
| `references/` | Propuesta, arquitectura y hallazgos. |
| `reports/figures/` | Figuras y evidencias. |
| `models/` | Modelos generados localmente. |

## Documentación del equipo

- [Cómo funcionan los stages y cómo integrar transformaciones](docs/flujo_datos.md).
- [Cómo registrar hallazgos](references/hallazgos/README.md).

Trabajar en ramas por tarea (`feat/`, `fix/`, `docs/`) e integrar mediante PR revisado. Coordinar una persona a la vez por notebook. Usar nombres `1.0-iniciales-descripcion.ipynb`; las contribuciones posteriores quedan en Git. La protección de `main` debe configurarse en GitHub.

## Estado de implementación

Esta guía corresponde al flujo desarrollado en la sesión: lectura local y stages `raw`, `structure`, `missing` y `common`. Una etapa puede estar vacía mientras se investigan sus tratamientos.

**Pendiente:** descarga automatizada, manifiesto de versiones, persistencia en Parquet, definición final de P1–P3, baselines y registro en MLflow. Se añadirán los comandos de entrenamiento cuando existan y estén comprobados.

Para la entrega, MLflow debe registrar al menos un run por pregunta, parámetros, versión de datos, métricas, artefactos y etiquetas de equipo, pregunta y commit. Cada integrante debe abrir un PR y revisar al menos otro.

Excluir del historial de Git los datos, modelos serializados, `.env`, `mlruns/` y `mlartifacts/`. Verificar el `.gitignore` antes de agregar archivos.

## Equipo

Número de equipo: por completar.

| Integrante | Iniciales | Rol o tarea |
|---|---|---|
| Por completar | — | — |
| Por completar | — | — |
| Por completar | — | — |
| Por completar | — | — |
