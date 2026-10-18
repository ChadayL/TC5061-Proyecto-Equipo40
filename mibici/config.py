from pathlib import Path

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
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
