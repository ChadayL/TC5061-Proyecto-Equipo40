# Desarrollo MiBici en contenedores

Jupyter y entrenamiento usan Python 3.10 y el mismo entorno administrado por uv.
El repositorio completo se monta en /workspace/mibici; los cambios se guardan en Windows.
El entorno Python Linux y la caché se conservan en volúmenes separados.
Al arrancar se ejecuta uv sync --locked. Se requiere internet para instalar dependencias.

Compose construye cada imagen desde el mismo Dockerfile. El servicio
ds_workspace selecciona la etapa jupyter, mlflow selecciona la etapa mlflow
y minio compila una versión fija del código oficial. La primera compilación
de MinIO descarga el compilador y las dependencias de Go.

## Preparación

Iniciar Docker Desktop. Configurar MINIO_ROOT_USER y MINIO_ROOT_PASSWORD en .env, sin publicarlas.
Cada servicio lee su propio archivo en configs/env/. En un clon nuevo, copiar
cada archivo .env.example a su nombre sin .example. Git ignora esas copias
locales y el .env de la raíz. Define MINIO_ROOT_USER y MINIO_ROOT_PASSWORD una sola vez en la raíz. Compose pasa esos valores a MinIO y los asigna a AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY en MLflow.

También puedes ajustar en el .env de la raíz los valores de Jupyter y MLflow:

~~~dotenv
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=TC5061-Proyecto-Equipo40
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_DEFAULT_REGION=us-east-1
~~~

Si se reutilizan modelos anteriores, conservar el nombre de proyecto Compose original con
docker compose -p NOMBRE_ANTERIOR en todos los comandos: cambiarlo crea otros volúmenes.
No ejecutar down -v ni levantar dos servidores MLflow sobre el mismo SQLite.

## Arranque

```powershell
docker compose config --quiet
docker compose up -d
docker compose logs ds_workspace
```

Abrir http://localhost:8888 usando el token mostrado por Jupyter.
MLflow está en http://localhost:5000 y MinIO en http://localhost:9001.
Compose arranca MinIO y ejecuta el servicio minio-init para preparar el bucket mlflow-artifacts. El inicializador termina con código 0 cuando el bucket está listo, y entonces arranca MLflow. Espera a que MLflow responda antes de entrenar.

## Entrenar desde la terminal de Jupyter

```bash
uv run python -c "import sys, mlflow, mibici; print(sys.version); print(mlflow.get_tracking_uri())"
uv run python -m mibici.dataset check
# Solo si faltan datos:
uv run mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e ingest
uv run mlflow run . --experiment-name "$MLFLOW_EXPERIMENT_NAME" --env-manager local -e train -P question=p1
```

Seleccionar el kernel Python 3 del contenedor. El experimento predeterminado es
TC5061-Proyecto-Equipo40; MLFLOW_EXPERIMENT_NAME permite cambiarlo desde .env.
Las dependencias de MiBici se fijan con uv.lock. Las imágenes base, uv y boto3
todavía no están fijadas por digest o versión en el Dockerfile de desarrollo.

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

## Colaboración

Trabajar en ramas por tarea (`feat/`, `fix/`, `docs/`) e integrar mediante PR revisado. Coordinar una persona a la vez por notebook y usar nombres `1.0-iniciales-descripcion.ipynb`.

Registrar los hallazgos con la [guía del equipo](../references/hallazgos/README.md). Mantener fuera de Git los datos de viajes, modelos serializados, `.env`, `mlruns/`, `mlflow.db` y `mlartifacts/`.
