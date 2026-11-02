from pathlib import Path

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates


DATA_FILE = Path(
    "data/interim/viajes/datos_abiertos_2025_02.parquet"
)


def load_sample(nrows: int = 10_000) -> pd.DataFrame:
    """Carga una muestra reproducible del dataset de viajes."""
    assert DATA_FILE.exists(), f"No existe el archivo: {DATA_FILE}"

    df = pd.read_parquet(DATA_FILE)

    if len(df) > nrows:
        df = df.sample(n=nrows, random_state=42)

    return df


def test_deepchecks_dataset_has_expected_columns():
    """Valida que las columnas principales del dataset estén presentes."""
    df = load_sample()

    expected_columns = {
        "Viaje_Id",
        "Usuario_Id",
        "Genero",
        "Año_de_nacimiento",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    }

    assert expected_columns.issubset(df.columns)


def test_deepchecks_no_duplicate_trips():
    """Utiliza DeepChecks para verificar viajes duplicados."""
    df = load_sample()

    dataset = Dataset(
        df,
        cat_features=["Genero"],
    )

    result = DataDuplicates().run(dataset)

    # DeepChecks devuelve la proporción de duplicados detectados.
    assert result.value == 0, (
        f"DeepChecks detectó duplicados: {result.value}"
    )


def test_deepchecks_critical_fields_not_null():
    """Valida integridad de campos críticos utilizados por los modelos."""
    df = load_sample()

    critical_columns = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    null_counts = df[critical_columns].isna().sum()

    assert null_counts.sum() == 0, (
        f"Se encontraron valores faltantes:\n{null_counts}"
    )


def test_training_sample_is_reproducible():
    """Comprueba reproducibilidad mediante una semilla fija."""
    df = pd.read_parquet(DATA_FILE)

    sample_1 = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    )

    sample_2 = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        sample_1.sort_index(),
        sample_2.sort_index(),
    )