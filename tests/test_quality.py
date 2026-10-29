import pandas as pd
import pytest

from mibici.validation import build_quality_frame, summarize_quality


def make_trips():
    return pd.DataFrame(
        {
            "Genero": ["M", None, "M"],
            "Origen_Id": [1, 2, 1],
            "Destino_Id": [2, 3, 2],
            "Inicio_del_viaje": pd.to_datetime(
                ["2025-01-01 08:00", "2025-01-02 08:00", "2025-01-01 08:00"]
            ),
            "Fin_del_viaje": pd.to_datetime(
                ["2025-01-01 08:10", "2025-01-02 08:30", "2025-01-01 08:10"]
            ),
            "Usuario_Id": ["private-a", "private-b", "private-a"],
        }
    )


def test_quality_frame_computes_duration_and_excludes_user_identifier():
    frame = build_quality_frame(make_trips())

    assert frame["duracion_minutos"].tolist() == [10, 30, 10]
    assert "Usuario_Id" not in frame.columns
    assert "Inicio_del_viaje" in frame.columns


def test_quality_frame_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_quality_frame(pd.DataFrame({"Genero": []}))


def test_quality_summary_reports_nulls_duplicates_and_types():
    frame = build_quality_frame(make_trips())
    summary = summarize_quality(frame)

    assert summary["rows"] == 3
    assert summary["duplicate_rows"] == 1
    assert summary["nulls_by_column"]["Genero"] == 1
    assert summary["dtypes"]["Origen_Id"] == "int64"
