from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Periodo completo del proyecto, ambos extremos incluidos.
DATA_START = "2024-10-01"
DATA_END = "2026-09-30"

# Partición temporal de P1, por días completos.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

#Edades mínimas y máximas para el año de nacimiento
MIN_AGE = 10
MAX_AGE = 100

# P2: riesgo cuando |salidas - llegadas| / capacidad alcanza este valor en un día.
# Regla heredada del equipo, pendiente de revisión.
P2_RISK_THRESHOLD = 0.50


def monthly_trip_files(start: str, end: str) -> tuple[str, ...]:
    """Genera los nombres mensuales del periodo configurado."""
    start_year, start_month = map(int, start.split("-")[:2])
    end_year, end_month = map(int, end.split("-")[:2])

    first_month = start_year * 12 + start_month - 1
    last_month = end_year * 12 + end_month - 1

    if first_month > last_month:
        raise ValueError("El inicio debe ser anterior al fin.")

    return tuple(
        f"datos_abiertos_{month // 12}_{month % 12 + 1:02d}.csv"
        for month in range(first_month, last_month + 1)
    )
    
TRIP_FILES = monthly_trip_files(DATA_START, DATA_END)  

STATIONS_FILE = "nomenclatura_2026_09.csv"

CSV_ENCODINGS = {
    "datos_abiertos_2024_10.csv": "cp1252",
    "datos_abiertos_2024_11.csv": "cp1252",
    "datos_abiertos_2024_12.csv": "cp1252",
    "datos_abiertos_2025_01.csv": "cp1252",
    "datos_abiertos_2025_02.csv": "cp1252",
    "datos_abiertos_2025_07.csv": "cp1252",
    "datos_abiertos_2026_01.csv": "cp1252",
}
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CSV_DATE_FORMATS = {
    "datos_abiertos_2024_10.csv": "%d/%m/%Y %H:%M",
}

# Fuente: https://mibici.net/es/datos-abiertos/
# Cada archivo vive en una carpeta numerada del portal que no sigue un patrón;
# la URL es SOURCE_BASE_URL/<carpeta>/<archivo>.
SOURCE_BASE_URL = "https://mibici.net/site/assets/files"

SOURCE_FOLDERS = {
    "datos_abiertos_2024_10.csv": 91758,
    "datos_abiertos_2024_11.csv": 93516,
    "datos_abiertos_2024_12.csv": 94962,
    "datos_abiertos_2025_01.csv": 97653,
    "datos_abiertos_2025_02.csv": 99879,
    "datos_abiertos_2025_03.csv": 101986,
    "datos_abiertos_2025_04.csv": 103813,
    "datos_abiertos_2025_05.csv": 105513,
    "datos_abiertos_2025_06.csv": 107088,
    "datos_abiertos_2025_07.csv": 108898,
    "datos_abiertos_2025_08.csv": 111021,
    "datos_abiertos_2025_09.csv": 113216,
    "datos_abiertos_2025_10.csv": 115104,
    "datos_abiertos_2025_11.csv": 116773,
    "datos_abiertos_2025_12.csv": 118153,
    "datos_abiertos_2026_01.csv": 120163,
    "datos_abiertos_2026_02.csv": 121761,
    "datos_abiertos_2026_03.csv": 123236,
    "datos_abiertos_2026_04.csv": 124927,
    "datos_abiertos_2026_05.csv": 125943,
    "datos_abiertos_2026_06.csv": 127734,
    "datos_abiertos_2026_07.csv": 129006,
    "datos_abiertos_2026_08.csv": 130362,
    "datos_abiertos_2026_09.csv": 131825,
    "nomenclatura_2026_09.csv": 1118,
}

# Salida de la ingesta: un Parquet por mes y el catálogo, ya normalizados.
TRIPS_PARQUET_DIR = INTERIM_DATA_DIR / "viajes"
STATIONS_PARQUET = INTERIM_DATA_DIR / "estaciones.parquet"

# Manifiesto versionado en Git: filas y sha256 de cada archivo descargado.
DATA_MANIFEST = PROJ_ROOT / "references" / "data_manifest.json"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "mibici-equipo40",
)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# Defaults de los modelos usados por la CLI y MLflow Project.
DEFAULT_MODELS = {"p1": "linear_regression", "p2": "random_forest", "p3": "kmeans"}
P1_FIT_INTERCEPT = True
P1_RIDGE_ALPHA = 1.0
P2_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}
P3_RANDOM_STATE = 42
P3_K_VALUES = [2, 3, 4, 5, 6]
P3_N_INIT = 10
