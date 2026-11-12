"""P1 monitoring: temporal data drift and paired A/B evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P1_RIDGE_ALPHA,
    REPORTS_DIR,
)
from mibici.dataset import get_data_version
from mibici.features import build_p1_features
from mibici.modeling.p1 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_evaluation_daily_data,
    build_model_pipeline,
    predict_departures,
    prepare_p1_data,
)
from mibici.modeling.tracking import DepartureModel
from mibici.pipeline import load_stage

app = typer.Typer(help=__doc__)

MODEL_NAME = "mibici-p1"
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REPORT_COLUMNS = FEATURE_COLUMNS + [TARGET]
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_BLOCK_DAYS = 7


def prepare_p1_monitoring_data() -> dict:
    """Build train, validation, and untouched later test data for P1.

    The test split is the simulated production period. The model candidate can
    use train+validation, while both models are evaluated on the same test rows.
    """
    prepared = prepare_p1_data()
    train = prepared["train"].copy()
    validation = prepared["validation"].copy()
    periods = prepared["periods"]
    test_start, test_end = periods["test"]

    known_stations = sorted(train["station_id"].unique())
    test_raw = load_stage("common", split="test")
    test_trips = test_raw["viajes"]
    test_daily = build_evaluation_daily_data(
        test_trips,
        station_ids=known_stations,
        start=test_start,
        end_exclusive=test_end,
    )

    # Preserve seven observed days immediately before test as lag context.
    history_start = test_start - pd.Timedelta(days=7)
    history = validation.loc[
        validation["station_id"].isin(known_stations)
        & validation["date"].ge(history_start)
        & validation["date"].lt(test_start),
        ["station_id", "date", TARGET],
    ]
    context = pd.concat([history, test_daily], ignore_index=True)
    test_features = (
        build_p1_features(context)
        .loc[lambda frame: frame["date"].ge(test_start) & frame["date"].lt(test_end)]
        .dropna(subset=FEATURE_COLUMNS + [TARGET])
        .reset_index(drop=True)
    )

    if train.empty or validation.empty or test_features.empty:
        raise ValueError("Train, validation y test deben contener filas para P1.")

    # Cast IDs/calendar to categorical strings only for Evidently's schema
    # inference. The estimator receives the original feature dtypes.
    reference = train[REPORT_COLUMNS + ["date"]].copy()
    current = test_features[REPORT_COLUMNS + ["date"]].copy()
    for frame in (reference, current):
        frame["station_id"] = frame["station_id"].astype("string")
        frame["day_of_week"] = frame["day_of_week"].astype("string")

    candidate_train = pd.concat([train, validation], ignore_index=True)
    return {
        "train": train,
        "validation": validation,
        "candidate_train": candidate_train,
        "test": test_features,
        "reference": reference,
        "current": current,
        "periods": periods,
        "summary": {
            "data_version": get_data_version(),
            "reference_start": str(periods["train"][0].date()),
            "reference_end_inclusive": str(
                (periods["train"][1] - pd.Timedelta(days=1)).date()
            ),
            "validation_start": str(periods["validation"][0].date()),
            "validation_end_inclusive": str(
                (periods["validation"][1] - pd.Timedelta(days=1)).date()
            ),
            "current_start": str(periods["test"][0].date()),
            "current_end_inclusive": str(
                (periods["test"][1] - pd.Timedelta(days=1)).date()
            ),
            "reference_rows": len(reference),
            "current_rows": len(current),
            "stations_evaluated": len(known_stations),
            "excluded_new_station_trips": int(
                (~test_trips["Origen_Id"].isin(known_stations)).sum()
            ),
            "evaluation_protocol": "one_step_daily_actual_lag_updates",
        },
    }


def calculate_regression_metrics(y_true, predictions) -> dict[str, float]:
    """Return consistent regression metrics for either A/B model."""
    return {
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def paired_daily_rmse_bootstrap(
    dates,
    y_true,
    incumbent_predictions,
    candidate_predictions,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired date-cluster bootstrap CI for candidate RMSE minus incumbent RMSE."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.normalize(),
            "incumbent_sq_error": np.square(np.asarray(y_true) - incumbent_predictions),
            "candidate_sq_error": np.square(np.asarray(y_true) - candidate_predictions),
        }
    )
    daily = frame.groupby("date", sort=True).agg(
        incumbent_sse=("incumbent_sq_error", "sum"),
        candidate_sse=("candidate_sq_error", "sum"),
        rows=("incumbent_sq_error", "size"),
    )
    if len(daily) < 2:
        raise ValueError("El intervalo bootstrap requiere al menos dos días de evaluación.")

    arrays = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    # Moving-block bootstrap retains short-term autocorrelation between days.
    block_size = min(BOOTSTRAP_BLOCK_DAYS, len(arrays))
    blocks_needed = int(np.ceil(len(arrays) / block_size))
    starts = rng.integers(0, len(arrays) - block_size + 1,
                          size=(replicates, blocks_needed))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets).reshape(replicates, -1)[:, :len(arrays)]
    sampled = arrays[indices].sum(axis=1)
    deltas = np.sqrt(sampled[:, 1] / sampled[:, 2]) - np.sqrt(sampled[:, 0] / sampled[:, 2])
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_rmse_candidate_minus_production": float(
            np.sqrt(arrays[:, 1].sum() / arrays[:, 2].sum())
            - np.sqrt(arrays[:, 0].sum() / arrays[:, 2].sum())
        ),
        "daily_cluster_bootstrap_95ci_low": float(low),
        "daily_cluster_bootstrap_95ci_high": float(high),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": f"moving_blocks_{block_size}_calendar_days; all stations per date remain together",
        "bootstrap_block_days": block_size,
    }


def extract_column_drift(report: dict) -> dict:
    """Extract Evidently 0.7 ValueDrift metrics from the serialized report."""
    found = {}
    for metric in report.get("metrics", []):
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column = config.get("column")
        score = metric.get("value")
        if column and isinstance(score, (int, float)):
            threshold = config.get("threshold", 0.1)
            found[column] = {
                "drift_detected": score >= threshold,
                "drift_score": float(score),
                "stattest_name": config.get("method"),
                "stattest_threshold": threshold,
            }
    return found


def summarize_data_quality(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "nulls_by_column": {name: int(value) for name, value in frame.isna().sum().items()},
        "dtypes": {name: str(value) for name, value in frame.dtypes.items()},
    }


def write_interpretation(path: Path, summary: dict, drift: dict, ab_result: dict) -> None:
    """Write a concise, data-backed interpretation alongside the HTML report."""
    affected = [
        name
        for name, result in drift.items()
        if result.get("drift_detected") is True
    ]
    implications = {
        "station_id": "cambió la mezcla de estaciones evaluadas; P1 solo incluye estaciones vistas durante entrenamiento",
        "day_of_week": "cambió el patrón semanal de demanda y puede sesgar la predicción por calendario",
        "departures_lag_1": "cambió la demanda del día previo, una señal inmediata para la predicción de un paso",
        "departures_lag_7": "cambió la demanda semanal comparable y puede indicar una nueva tendencia",
        "departures_mean_7": "cambió el nivel de demanda reciente suavizado de cada estación",
        "departures": "cambió el volumen observado que el modelo intenta predecir; es drift de la variable objetivo",
    }
    lines = [
        "# Interpretación de monitoreo P1",
        "",
        f"- Datos: `{summary['data_version']}`.",
        f"- Referencia (train): {summary['reference_start']} a {summary['reference_end_inclusive']}.",
        f"- Periodo actual simulado (test): {summary['current_start']} a {summary['current_end_inclusive']}.",
        f"- Filas: {summary['reference_rows']:,} de referencia y {summary['current_rows']:,} actuales.",
        f"- Estaciones evaluadas: {summary['stations_evaluated']}; viajes de estaciones fuera del alcance excluidos: {summary['excluded_new_station_trips']:,}.",
        "- El reporte Evidently incluye drift por columna y calidad de los dos conjuntos.",
        "",
        "## Variables con drift detectado por Evidently",
        "",
    ]
    if affected:
        for name in affected:
            details = drift[name]
            implication = implications.get(name, "cambió su distribución frente al periodo de entrenamiento")
            lines.append(
                f"- **{name}**: score `{details.get('drift_score')}`, prueba `{details.get('stattest_name')}`; {implication}."
            )
    else:
        lines.append("Ninguna columna cruzó el umbral configurado (0.10). Scores por variable:")
        for name, details in drift.items():
            lines.append(
                f"- `{name}`: {details.get('drift_score', float('nan')):.4f} "
                f"({details.get('stattest_name')}, umbral {details.get('stattest_threshold')})."
            )
    quality = ab_result.get("data_quality", {})
    lines.extend(
        [
            "",
            "El drift estadístico señala un cambio de distribución; por sí solo no prueba degradación del modelo. La decisión se basa además en el A/B sobre las mismas filas actuales y en el intervalo bootstrap pareado por día.",
            "",
            "## Comparación A/B sobre el mismo periodo actual",
            "",
            f"- Calidad de referencia: {quality.get('reference', {}).get('rows')} filas; {quality.get('reference', {}).get('duplicate_rows')} duplicadas; {sum(quality.get('reference', {}).get('nulls_by_column', {}).values())} valores nulos.",
            f"- Calidad del periodo actual: {quality.get('current', {}).get('rows')} filas; {quality.get('current', {}).get('duplicate_rows')} duplicadas; {sum(quality.get('current', {}).get('nulls_by_column', {}).values())} valores nulos.",
            "",
            f"- Versión vigente: `{ab_result['production_model_version']}`; RMSE {ab_result['production_metrics']['rmse']:.3f}, MAE {ab_result['production_metrics']['mae']:.3f}, R² {ab_result['production_metrics']['r2']:.3f}.",
            f"- Candidata: `{ab_result['candidate_model_version']}` ({ab_result['candidate_model']}); RMSE {ab_result['candidate_metrics']['rmse']:.3f}, MAE {ab_result['candidate_metrics']['mae']:.3f}, R² {ab_result['candidate_metrics']['r2']:.3f}.",
            f"- Δ RMSE candidata − vigente: {ab_result['delta_rmse_candidate_minus_production']:.3f}; IC 95% por remuestreo diario [{ab_result['daily_cluster_bootstrap_95ci_low']:.3f}, {ab_result['daily_cluster_bootstrap_95ci_high']:.3f}].",
            f"- Recomendación: **{ab_result['recommendation']}**.",
            "",
            "El remuestreo conserva juntas todas las estaciones de cada fecha y compara errores pareados. Si el intervalo contiene cero, la evidencia no separa con claridad a los modelos y se recomienda mantener la versión vigente.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_registry_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def run_monitoring(candidate_model: str = "ridge", alpha: float = P1_RIDGE_ALPHA) -> dict:
    """Run P1 drift report and register an A/B candidate without moving production."""
    if candidate_model not in {"linear_regression", "ridge"}:
        raise ValueError("candidate_model debe ser linear_regression o ridge")
    if alpha <= 0:
        raise ValueError("alpha debe ser positivo")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not model_versions:
        raise RuntimeError(
            f"No existe {MODEL_NAME} en el Registry. Registra/promueve primero el modelo P1."
        )
    try:
        production_version = client.get_model_version_by_alias(MODEL_NAME, "production")
    except MlflowException as exc:
        raise RuntimeError(
            f"El modelo {MODEL_NAME} no tiene alias production. Asigna primero el modelo vigente."
        ) from exc

    datasets = prepare_p1_monitoring_data()
    train = datasets["candidate_train"]
    test = datasets["test"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET].to_numpy(dtype=float)

    production_uri = f"models:/{MODEL_NAME}@production"
    production = mlflow.pyfunc.load_model(production_uri)
    incumbent_predictions = np.asarray(production.predict(X_test), dtype=float).reshape(-1)

    candidate = build_model_pipeline(
        model_name=candidate_model,
        fit_intercept=True,
        alpha=alpha,
    )
    candidate.fit(train[FEATURE_COLUMNS], train[TARGET])
    candidate_predictions = np.asarray(
        predict_departures(candidate, X_test), dtype=float
    ).reshape(-1)

    production_metrics = calculate_regression_metrics(y_test, incumbent_predictions)
    candidate_metrics = calculate_regression_metrics(y_test, candidate_predictions)
    bootstrap = paired_daily_rmse_bootstrap(
        test["date"], y_test, incumbent_predictions, candidate_predictions
    )
    low = bootstrap["daily_cluster_bootstrap_95ci_low"]
    high = bootstrap["daily_cluster_bootstrap_95ci_high"]
    if high < 0:
        recommendation = "candidate"
    elif low > 0:
        recommendation = "production"
    else:
        recommendation = "production (resultado inconcluso)"

    drift_frame = datasets["current"]
    reference_frame = datasets["reference"]
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as exc:
        raise RuntimeError(
            "Instala el extra de monitoreo: uv sync --locked --extra monitoring"
        ) from exc

    drift_report = Report(
        metrics=[
            DataDriftPreset(columns=REPORT_COLUMNS),
            DataSummaryPreset(columns=REPORT_COLUMNS + ["date"]),
        ],
        include_tests=True,
        metadata={
            "project": "MiBici Equipo 40",
            "question": "P1",
            "reference_period": (
                f"{datasets['summary']['reference_start']}"
                f"/{datasets['summary']['reference_end_inclusive']}"
            ),
            "current_period": (
                f"{datasets['summary']['current_start']}"
                f"/{datasets['summary']['current_end_inclusive']}"
            ),
            "data_version": datasets["summary"]["data_version"],
        },
    )
    snapshot = drift_report.run(
        current_data=drift_frame,
        reference_data=reference_frame,
        name="p1_train_vs_later_test",
    )
    report_data = json.loads(snapshot.json())
    drift_by_column = extract_column_drift(report_data)

    output_dir = REPORTS_DIR / "monitoring" / "p1"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_html = output_dir / "evidently_drift_quality.html"
    report_json = output_dir / "evidently_drift_quality.json"
    reference_quality_html = output_dir / "calidad_referencia.html"
    current_quality_html = output_dir / "calidad_actual.html"
    interpretation_path = output_dir / "interpretacion_ab.md"
    predictions_path = output_dir / "predicciones_ab.csv"
    result_path = output_dir / "resultado_ab.json"
    snapshot.save_html(str(report_html))
    report_json.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    for frame, path, name in (
        (reference_frame, reference_quality_html, "calidad_referencia"),
        (drift_frame, current_quality_html, "calidad_actual"),
    ):
        quality_snapshot = Report(
            metrics=[DataSummaryPreset(columns=REPORT_COLUMNS + ["date"])],
            include_tests=True,
        ).run(current_data=frame, name=name)
        quality_snapshot.save_html(str(path))

    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento MLflow {MLFLOW_EXPERIMENT_NAME!r}")
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    with mlflow.start_run(run_name="p1_evidently_ab_monitoring"):
        candidate_info = mlflow.pyfunc.log_model(
            name="p1_ab_candidate",
            python_model=DepartureModel(candidate),
            input_example=X_test.head(5),
        )
        _ensure_registry_model(client, MODEL_NAME)
        version = mlflow.register_model(candidate_info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "ab-candidate", version.version)

        result = {
            **datasets["summary"],
            "reference_model": production_uri,
            "production_model_version": str(production_version.version),
            "production_training_data_version": production_version.tags.get("data_version"),
            "production_training_commit": production_version.tags.get("commit"),
            "candidate_model": candidate_model,
            "candidate_alpha": alpha if candidate_model == "ridge" else None,
            "candidate_model_version": str(version.version),
            "candidate_model_alias": "ab-candidate",
            "production_metrics": production_metrics,
            "candidate_metrics": candidate_metrics,
            **bootstrap,
            "recommendation": recommendation,
            "drift_by_column": drift_by_column,
            "data_quality": {
                "reference": summarize_data_quality(reference_frame),
                "current": summarize_data_quality(drift_frame),
            },
            "reports": [
                report_html.name,
                report_json.name,
                reference_quality_html.name,
                current_quality_html.name,
            ],
        }
        write_interpretation(interpretation_path, datasets["summary"], drift_by_column, result)
        pd.DataFrame(
            {
                "date": test["date"].to_numpy(),
                "station_id": test["station_id"].to_numpy(),
                "actual_departures": y_test,
                "production_prediction": incumbent_predictions,
                "candidate_prediction": candidate_predictions,
            }
        ).to_csv(predictions_path, index=False)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        mlflow.log_params(
            {
                "question": "p1",
                "candidate_model": candidate_model,
                "candidate_alpha": alpha if candidate_model == "ridge" else "not_applicable",
                "candidate_train_rows": len(train),
                "test_rows": len(test),
                "data_version": datasets["summary"]["data_version"],
                "production_model_version": production_version.version,
                "candidate_model_version": version.version,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            }
        )
        mlflow.log_metrics(
            {
                "production_test_rmse": production_metrics["rmse"],
                "production_test_mae": production_metrics["mae"],
                "production_test_r2": production_metrics["r2"],
                "candidate_test_rmse": candidate_metrics["rmse"],
                "candidate_test_mae": candidate_metrics["mae"],
                "candidate_test_r2": candidate_metrics["r2"],
                "delta_test_rmse_candidate_minus_production": bootstrap[
                    "delta_rmse_candidate_minus_production"
                ],
                "rmse_delta_ci95_low": low,
                "rmse_delta_ci95_high": high,
                "drifted_columns": sum(
                    detail.get("drift_detected") is True
                    for detail in drift_by_column.values()
                ),
            }
        )
        mlflow.set_tags(
            {
                "question": "p1",
                "monitoring_tool": "evidently",
                "data_version": datasets["summary"]["data_version"],
                "production_model_version": production_version.version,
                "candidate_model_version": version.version,
                "recommendation": recommendation,
                "reference_period": (
                    f"{datasets['summary']['reference_start']}"
                    f"/{datasets['summary']['reference_end_inclusive']}"
                ),
                "current_period": (
                    f"{datasets['summary']['current_start']}"
                    f"/{datasets['summary']['current_end_inclusive']}"
                ),
            }
        )
        mlflow.log_artifacts(str(output_dir), artifact_path="monitoring/p1")
        mlflow.log_input(
            mlflow.data.from_pandas(
                train[REPORT_COLUMNS],
                source=datasets["summary"]["data_version"],
                name="p1_reference_train",
            ),
            context="reference",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                test[REPORT_COLUMNS],
                source=datasets["summary"]["data_version"],
                name="p1_current_test",
                targets=TARGET,
            ),
            context="current_evaluation",
        )

    typer.echo(f"Reporte Evidently: {report_html}")
    typer.echo(f"Interpretación y A/B: {interpretation_path}")
    typer.echo(f"Producción: versión {production_version.version}, métricas {production_metrics}")
    typer.echo(f"Candidata: versión {version.version}, métricas {candidate_metrics}")
    typer.echo(
        f"Delta RMSE candidata - produccion: {bootstrap['delta_rmse_candidate_minus_production']:.3f} "
        f"(IC 95% {low:.3f}, {high:.3f})"
    )
    typer.echo(f"Recomendación: {recommendation}; alias ab-candidate asignado.")
    typer.echo("El alias production se conserva sin cambios.")
    return result


@app.command()
def main(
    candidate_model: str = typer.Option("ridge", "--candidate-model"),
    alpha: float = typer.Option(P1_RIDGE_ALPHA, "--alpha", min=0.000001),
):
    """Genera Evidently y compara producción contra una candidata reentrenada."""
    run_monitoring(candidate_model=candidate_model, alpha=alpha)


if __name__ == "__main__":
    app()
