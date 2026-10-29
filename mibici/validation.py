"""Calidad de datos con Deepchecks y perfiles agregados sin exponer viajes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici.dataset import get_data_version, load_raw

QUALITY_COLUMNS = ["Inicio_del_viaje", "Fin_del_viaje", "Genero", "Origen_Id", "Destino_Id", "duracion_minutos"]


def build_quality_frame(trips: pd.DataFrame) -> pd.DataFrame:
    """Prepara variables para validar sin IDs de usuario; conserva fechas para detectar duplicados exactos."""
    required = {"Genero", "Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje"}
    missing = required - set(trips.columns)
    if missing:
        raise ValueError(f"Faltan columnas de viajes para validar: {sorted(missing)}")

    result = trips[["Inicio_del_viaje", "Fin_del_viaje", "Genero", "Origen_Id", "Destino_Id"]].copy()
    result["duracion_minutos"] = (
        (trips["Fin_del_viaje"] - trips["Inicio_del_viaje"]).dt.total_seconds() / 60
    )
    return result[QUALITY_COLUMNS]


def summarize_quality(data: pd.DataFrame) -> dict:
    """Devuelve conteos agregados de nulos, duplicados y tipos, sin valores de filas."""
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "duplicate_rows": int(data.duplicated().sum()),
        "nulls_by_column": {name: int(count) for name, count in data.isna().sum().items()},
        "dtypes": {name: str(dtype) for name, dtype in data.dtypes.items()},
    }


def run_deepchecks(train: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> dict:
    """Ejecuta integridad y drift de Deepchecks y guarda reportes HTML locales."""
    try:
        # Deepchecks 0.19 usa np.Inf, eliminado en NumPy 2; conservar compatibilidad.
        if not hasattr(np, "Inf"):
            np.Inf = np.inf
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity, train_test_validation
    except ImportError as exc:
        raise RuntimeError(
            "Deepchecks no pudo importarse. Instala el grupo quality: "
            "uv sync --locked --extra quality"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = Dataset(train, cat_features=["Genero"], datetime_name="Inicio_del_viaje")
    validation_dataset = Dataset(validation, cat_features=["Genero"], datetime_name="Inicio_del_viaje")
    train_result = data_integrity().run(train_dataset, with_display=False)
    split_result = train_test_validation(
        random_state=42, n_samples=min(len(train), len(validation))
    ).run(train_dataset, validation_dataset, with_display=False)
    train_result.save_as_html(str(output_dir / "integridad_train.html"))
    split_result.save_as_html(str(output_dir / "comparacion_train_validation.html"))

    summary = {
        "data_version": get_data_version(),
        "train": summarize_quality(train),
        "validation": summarize_quality(validation),
        "reports": ["integridad_train.html", "comparacion_train_validation.html"],
    }
    (output_dir / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida calidad de datos MiBici con Deepchecks")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality"))
    args = parser.parse_args()
    if args.sample_rows < 1:
        parser.error("--sample-rows debe ser positivo")

    train = build_quality_frame(load_raw(nrows=args.sample_rows, split="train")["viajes"])
    validation = build_quality_frame(
        load_raw(nrows=args.sample_rows, split="validation")["viajes"]
    )
    summary = run_deepchecks(train, validation, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reportes guardados en {args.output.resolve()}")


if __name__ == "__main__":
    main()
