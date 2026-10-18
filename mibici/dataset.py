import math

import pandas as pd
from loguru import logger
import typer

from mibici.config import (
    DATA_END,
    DATA_START,
    RAW_DATA_DIR,
    STATIONS_FILE,
    TEST_RATIO,
    TRAIN_RATIO,
    TRIP_FILES,
    VALIDATION_RATIO,
    CSV_ENCODINGS,
    CSV_DATE_FORMATS,
    DEFAULT_DATE_FORMAT,
)

app = typer.Typer()


def load_csv(filename: str, nrows: int | None = None) -> pd.DataFrame:
    """Lee un CSV de data/raw."""
    return pd.read_csv(
        RAW_DATA_DIR / filename,
        encoding="utf-8",
        nrows=nrows,
    )


def check_required_files():
    """Informa todos los archivos configurados que faltan."""
    filenames = list(TRIP_FILES) + [STATIONS_FILE]

    missing_files = [
        filename for filename in filenames if not (RAW_DATA_DIR / filename).is_file()
    ]

    if missing_files:
        details = "\n".join(f"  - {name}" for name in missing_files)
        raise FileNotFoundError(f"Faltan archivos en {RAW_DATA_DIR}:\n{details}")


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

        encoding = CSV_ENCODINGS.get(filename, "utf-8-sig")
        with pd.read_csv(
            RAW_DATA_DIR / filename,
            encoding=encoding,
            chunksize=100_000,
        ) as reader:
            for chunk in reader:
                # Interpretamos la fecha para separar, sin reemplazar
                # todavía la columna original de los datos raw.

                date_format = CSV_DATE_FORMATS.get(filename, DEFAULT_DATE_FORMAT)

                start_dates = pd.to_datetime(
                    chunk["Inicio_del_viaje"],
                    format=date_format,
                    errors="raise",
                )

                if start_dates.isna().any():
                    raise ValueError(
                        f"{filename}: hay fechas de inicio faltantes; "
                        "no se puede asignar su partición."
                    )

                selected = chunk.loc[
                    start_dates.ge(start) & start_dates.lt(end_exclusive)
                ]

                if nrows is not None:
                    selected = selected.head(nrows - loaded_rows)

                if not selected.empty:
                    tables.append(selected.copy())
                    loaded_rows += len(selected)

                if nrows is not None and loaded_rows >= nrows:
                    break

    if not tables:
        raise ValueError(f"No se encontraron viajes para la partición {split!r}.")

    trips = pd.concat(tables, ignore_index=True)
    stations = load_csv(STATIONS_FILE)

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
def main(split: str = "train", nrows: int = 5):
    """Comprueba la carga con una muestra de la partición."""
    data = load_raw(split=split, nrows=nrows)

    logger.info(f"Viajes en la muestra: {len(data['viajes']):,}")
    logger.info(f"Estaciones en el catálogo: {len(data['estaciones']):,}")
    logger.success("Archivos leídos correctamente.")


if __name__ == "__main__":
    app()
