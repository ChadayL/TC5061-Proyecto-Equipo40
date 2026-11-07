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
locales y el .env de la raíz, donde se definen las credenciales.
Si se reutilizan modelos anteriores, conservar el nombre de proyecto Compose original con
docker compose -p NOMBRE_ANTERIOR en todos los comandos: cambiarlo crea otros volúmenes.
No ejecutar down -v ni levantar dos servidores MLflow sobre el mismo SQLite.

## Arranque

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose logs ds_workspace
```

Abrir http://localhost:8888 usando el token mostrado por Jupyter.
MLflow está en http://localhost:5000 y MinIO en http://localhost:9001.
En una instalación nueva, crear el bucket mlflow-artifacts en MinIO antes de registrar artefactos.
depends_on ordena el arranque; esperar a que MLflow responda antes de entrenar.

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
