import pandas as pd
from loguru import logger
import typer

from mibici.config import RAW_DATA_DIR, STATIONS_FILE, TRIP_FILES

app = typer.Typer()


def load_csv(filename: str, nrows: int | None = None) -> pd.DataFrame:
    """Lee un CSV desde data/raw sin modificar el archivo."""
    return pd.read_csv(
        RAW_DATA_DIR / filename,
        encoding="utf-8",
        nrows=nrows,
    )


def load_raw(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Carga los meses seleccionados y el catálogo de estaciones."""
    viajes = pd.concat(
        [load_csv(filename, nrows=nrows) for filename in TRIP_FILES],
        ignore_index=True,
    )

    estaciones = load_csv(STATIONS_FILE)

    return {
        "viajes": viajes,
        "estaciones": estaciones,
    }


@app.command()
def main():
    """Comprueba la lectura de los archivos con una muestra."""
    datos = load_raw(nrows=5)

    logger.info(f"Viajes en la muestra: {len(datos['viajes'])}")
    logger.info(f"Estaciones en el catálogo: {len(datos['estaciones'])}")
    logger.success("Archivos leídos correctamente.")


if __name__ == "__main__":
    app()