import pandas as pd
import pytest

from mibici.data_preparation.quality_fixes import (
    remove_duplicate_trips,
    replace_invalid_birth_years,
)
from mibici.data_preparation.structure import convert_dates, normalize_columns

def test_normalize_columns_removes_surrounding_spaces():
    frame = pd.DataFrame([[1, "M"]], columns=[" Origen_Id ", "Genero "])

    result = normalize_columns(frame)

    assert result.columns.tolist() == ["Origen_Id", "Genero"]


def test_convert_dates_accepts_both_project_formats():
    trips = pd.DataFrame(
        {
            "Inicio_del_viaje": [
                "31/10/2024 23:30",
                "2025-01-01 08:00:00",
            ],
            "Fin_del_viaje": [
                "01/11/2024 00:00",
                "2025-01-01 08:30:00",
            ],
        }
    )

    result = convert_dates(trips)

    assert result.loc[0, "Inicio_del_viaje"] == pd.Timestamp("2024-10-31 23:30")
    assert result.loc[1, "Fin_del_viaje"] == pd.Timestamp("2025-01-01 08:30")


def test_convert_dates_rejects_invalid_date():
    trips = pd.DataFrame(
        {
            "Inicio_del_viaje": ["31/02/2025 08:00"],
            "Fin_del_viaje": ["31/02/2025 08:30"],
        }
    )

    with pytest.raises(ValueError):
        convert_dates(trips)


def test_remove_duplicate_trips_keeps_one_copy():
    trips = pd.DataFrame({"trip_id": [1, 1, 2]})

    result = remove_duplicate_trips(trips)

    assert result["trip_id"].tolist() == [1, 2]


def test_replace_invalid_birth_years_preserves_age_boundaries():
    trips = pd.DataFrame(
        {
            "Año_de_nacimiento": pd.Series(
                [2007, 1965, 2008, 1964, 0, pd.NA], dtype="Int64"
            ),
            "Inicio_del_viaje": pd.to_datetime(["2025-01-01"] * 6),
        }
    )

    result = replace_invalid_birth_years(trips, min_age=18, max_age=60)
    birth_years = result["Año_de_nacimiento"]

    assert birth_years.iloc[0] == 2007
    assert birth_years.iloc[1] == 1965
    assert birth_years.iloc[2:].isna().all()
