import hashlib
import json
import math
import urllib.request

import pandas as pd
from loguru import logger
import typer

from mibici.config import (
    CSV_DATE_FORMATS,
    CSV_ENCODINGS,
    DATA_END,
    DATA_MANIFEST,
    DATA_START,
    DEFAULT_DATE_FORMAT,
    PROJ_ROOT,
    RAW_DATA_DIR,
    SOURCE_BASE_URL,
    SOURCE_FOLDERS,
    STATIONS_FILE,
    STATIONS_PARQUET,
    TEST_RATIO,
    TRAIN_RATIO,
    TRIP_FILES,
    TRIPS_PARQUET_DIR,
    VALIDATION_RATIO,
)

app = typer.Typer()

# El portal rechaza (HTTP 406) las peticiones sin encabezados de navegador.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}


# ============================================================
# INGESTA
# ============================================================

def download(filename):
    """Descarga un archivo del portal a data/raw, si todavía no existe."""
    path = RAW_DATA_DIR / filename

    if not path.is_file():
        url = f"{SOURCE_BASE_URL}/{SOURCE_FOLDERS[filename]}/{filename}"
        logger.info(f"Descargando {url}")
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
            path.write_bytes(response.read())

    return path


def read_trips(filename):
    """Lee un mes con su codificación, quita espacios de las columnas y convierte las fechas."""
    trips = pd.read_csv(
        RAW_DATA_DIR / filename,
        encoding=CSV_ENCODINGS.get(filename, "utf-8-sig"),
    )
    trips.columns = trips.columns.str.strip()

    date_format = CSV_DATE_FORMATS.get(filename, DEFAULT_DATE_FORMAT)
    for column in ["Inicio_del_viaje", "Fin_del_viaje"]:
        trips[column] = pd.to_datetime(trips[column], format=date_format)

    return trips


def trips_parquet(filename):
    return TRIPS_PARQUET_DIR / filename.replace(".csv", ".parquet")


@app.command()
def ingest():
    """Descarga los archivos configurados, los guarda en Parquet y escribe el manifiesto."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRIPS_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    for filename in [*TRIP_FILES, STATIONS_FILE]:
        path = download(filename)

        if filename == STATIONS_FILE:
            data = pd.read_csv(path, encoding="utf-8")
            data.to_parquet(STATIONS_PARQUET, index=False)
        else:
            data = read_trips(filename)
            data.to_parquet(trips_parquet(filename), index=False)

        files.append({
            "file": filename,
            "rows": len(data),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        logger.info(f"{filename}: {len(data):,} filas")

    # La versión cambia si cambia cualquier archivo descargado.
    data_version = hashlib.sha256(json.dumps(files).encode()).hexdigest()[:12]

    manifest = {
        "source": SOURCE_BASE_URL,
        "date_range": [DATA_START, DATA_END],
        "data_version": data_version,
        "files": files,
    }
    DATA_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.success(f"Ingesta completa. Versión de los datos: {data_version}")


def get_data_version():
    """Versión de los datos del manifiesto; se registra en cada run de MLflow."""
    return json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))["data_version"]


# ============================================================
# CARGA POR PARTICIÓN
# ============================================================

def check_required_files():
    """Informa todos los Parquet de la ingesta que faltan."""
    paths = [trips_parquet(filename) for filename in TRIP_FILES] + [STATIONS_PARQUET]
    missing = [path for path in paths if not path.is_file()]

    if missing:
        details = "\n".join(f"  - {path.relative_to(PROJ_ROOT)}" for path in missing)
        raise FileNotFoundError(
            f"Faltan datos ingeridos:\n{details}\nEjecuta: python -m mibici.dataset ingest"
        )


def get_split_periods():
    """Define intervalos temporales con inicio incluido y fin excluido."""
    ratios = [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO]

    if not all(math.isfinite(ratio) and ratio > 0 for ratio in ratios):
        raise ValueError("Las proporciones deben ser positivas y finitas.")

    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Las proporciones deben sumar 1.")

    start = pd.Timestamp(DATA_START)
    end = pd.Timestamp(DATA_END)

    if start > end:
        raise ValueError("DATA_START debe ser anterior o igual a DATA_END.")

    calendar = pd.date_range(start, end, freq="D")

    train_days = int(len(calendar) * TRAIN_RATIO)
    validation_days = int(len(calendar) * VALIDATION_RATIO)
    test_days = len(calendar) - train_days - validation_days

    if min(train_days, validation_days, test_days) < 1:
        raise ValueError("Cada partición debe contener al menos un día.")

    validation_start = calendar[train_days]
    test_start = calendar[train_days + validation_days]
    end_exclusive = end + pd.Timedelta(days=1)

    return {
        "train": (start, validation_start),
        "validation": (validation_start, test_start),
        "test": (test_start, end_exclusive),
    }


def load_raw(
    nrows: int | None = None,
    split: str = "train",
) -> dict[str, pd.DataFrame]:
    """Carga los viajes de una partición (por Inicio_del_viaje) y el catálogo ingeridos."""
    periods = get_split_periods()

    if split not in periods:
        raise ValueError(f"Partición desconocida: {split!r}. Opciones: {list(periods)}")

    if nrows is not None and nrows < 1:
        raise ValueError("nrows debe ser positivo o None.")

    check_required_files()

    start, end_exclusive = periods[split]

    selected_months = set(
        pd.period_range(
            start,
            end_exclusive - pd.Timedelta(days=1),
            freq="M",
        ).strftime("%Y_%m")
    )

    selected_files = [
        filename
        for filename in TRIP_FILES
        if filename.removeprefix("datos_abiertos_").removesuffix(".csv")
        in selected_months
    ]

    tables = []
    # nrows limita el total de la partición, no cada archivo.
    loaded_rows = 0

    for filename in selected_files:
        if nrows is not None and loaded_rows >= nrows:
            break

        trips = pd.read_parquet(trips_parquet(filename))
        start_dates = trips["Inicio_del_viaje"]

        if start_dates.isna().any():
            raise ValueError(
                f"{filename}: hay fechas de inicio faltantes; "
                "no se puede asignar su partición."
            )

        selected = trips.loc[start_dates.ge(start) & start_dates.lt(end_exclusive)]

        if nrows is not None:
            selected = selected.head(nrows - loaded_rows)

        if not selected.empty:
            tables.append(selected)
            loaded_rows += len(selected)

    if not tables:
        raise ValueError(f"No se encontraron viajes para la partición {split!r}.")

    trips = pd.concat(tables, ignore_index=True)
    stations = pd.read_parquet(STATIONS_PARQUET)

    logger.info(
        f"Partición {split}: {start.date()} a "
        f"{(end_exclusive - pd.Timedelta(days=1)).date()}"
    )
    logger.info(f"Viajes cargados: {len(trips):,}")

    return {
        "viajes": trips,
        "estaciones": stations,
    }


@app.command()
def check(split: str = "train", nrows: int = 5):
    """Comprueba la carga con una muestra de la partición."""
    data = load_raw(split=split, nrows=nrows)

    logger.info(f"Viajes en la muestra: {len(data['viajes']):,}")
    logger.info(f"Estaciones en el catálogo: {len(data['estaciones']):,}")
    logger.info(f"Versión de los datos: {get_data_version()}")
    logger.success("Archivos leídos correctamente.")


if __name__ == "__main__":
    app()
