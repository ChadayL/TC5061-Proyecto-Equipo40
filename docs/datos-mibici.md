# Datos de MiBici

## Ingesta de datos

Fuente: [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Ejecuta la ingesta desde la raíz del repositorio dentro del [contenedor de trabajo](jupyter-contenedor.md).

```bash
uv run mlflow run --experiment-name "$MLFLOW_EXPERIMENT_NAME" . --env-manager local -e ingest
```

1. **Descarga** a `data/raw/` los 24 meses de viajes (`datos_abiertos_2024_10.csv` … `datos_abiertos_2026_09.csv`) y el catálogo de estaciones `nomenclatura_2026_10.csv`. Omite los archivos que ya existen.
2. **Normaliza** cada archivo: su codificación (UTF-8 o cp1252), nombres de columna sin espacios y fechas en sus dos formatos (`dd/mm/aaaa HH:MM` en octubre de 2024, `aaaa-mm-dd HH:MM:SS` en el resto) a `datetime`.
3. **Guarda** un Parquet por mes en `data/interim/viajes/` y el catálogo en `data/interim/estaciones.parquet`.
4. **Versiona** los datos en [`references/data_manifest.json`](../references/data_manifest.json), que sí se sube a Git: número de registros y sha256 de cada archivo, más una `data_version` que se registra en cada run de MLflow. Si MiBici cambia algún archivo, cambian su hash y la versión, y `git diff` del manifiesto lo muestra.

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

Carga una muestra de cinco viajes de train, el catálogo completo disponible y muestra la versión de los datos. Opciones: `--split` y `--nrows`.

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

Consulta el [flujo de datos](flujo_datos.md) para integrar otras transformaciones.

## Aspectos pendientes de los datos

- **Pendiente en P2:** fórmula y umbral de riesgo, y la inclusión de estaciones sin actividad.
- **Pendiente en P3:** normalizar por días activos y definir el uso del catálogo de estaciones.
- **Otros:** límites de edad (H-co-006), elegibilidad de estaciones en el calendario de P1 (H-co-010).
