from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import typer

from mibici.config import DEFAULT_MODELS, P1_FIT_INTERCEPT, P1_RIDGE_ALPHA, P3_N_INIT
from mibici.dataset import get_data_version
from mibici.modeling import p2, p3
from mibici.modeling.p1 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model_pipeline,
    predict_departures,
    prepare_p1_data,
)
from mibici.modeling.tracking import (
    DepartureModel,
    EstimatorModel,
    log_model_evaluation,
)
from mibici.plots import (
    plot_classification,
    plot_cluster_profiles,
    plot_regression,
    plot_station_clusters,
)

app = typer.Typer()


def run_p1(
    model_name: str, fit_intercept: bool, alpha=P1_RIDGE_ALPHA,
    stage_of_work="candidate", **_options,
):
    if model_name not in {"linear_regression", "ridge"}:
        raise ValueError(f"Modelo no disponible para P1: {model_name!r}")
    datasets = prepare_p1_data()

    train = datasets["train"]
    validation = datasets["validation"]
    periods = datasets["periods"]

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET]

    model = build_model_pipeline(
        fit_intercept=fit_intercept, model_name=model_name, alpha=alpha,
    )
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started

    started = perf_counter()
    predictions = predict_departures(model, X_validation)
    predict_seconds = perf_counter() - started

    baseline_predictions = validation["departures_lag_1"].clip(lower=0)
    metrics = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "validation_mae": mean_absolute_error(y_validation, predictions),
        "validation_rmse": mean_squared_error(y_validation, predictions) ** 0.5,
        "baseline_validation_mae": mean_absolute_error(y_validation, baseline_predictions),
        "baseline_validation_rmse": mean_squared_error(y_validation, baseline_predictions) ** 0.5,
    }

    params = {
        "data_version": get_data_version(),
        "model": model_name,
        "fit_intercept": fit_intercept,
        "prediction_floor": 0,
        "horizon_days": 1,
        "evaluation_protocol": "daily_one_step",
        "station_scope": "stations_with_training_history",
        "train_start": str(periods["train"][0].date()),
        "train_end_exclusive": str(periods["train"][1].date()),
        "validation_start": str(periods["validation"][0].date()),
        "validation_end_exclusive": str(periods["validation"][1].date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "features": ",".join(feature_columns),
        **datasets["summary"],
    }
    if model_name == "ridge":
        params["alpha"] = alpha
        params["solver"] = "lsqr"

    wrapped_model = DepartureModel(model)
    example = X_train.head(5)

    run_id, model_uri = log_model_evaluation(
        run_name=f"p1_{model_name}",
        python_model=wrapped_model,
        input_example=example,
        output_example=wrapped_model.predict(None, example),
        params=params,
        metrics=metrics,
        tags={
            "problem": "p1",
            "run_type": "model",
            "execution": "cli",
        },
        figures={"regression.png": plot_regression(y_validation, predictions)},
        training_data=train[feature_columns + [TARGET]],
        target=TARGET,
        stage_of_work=stage_of_work,
    )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Modelo: {model_uri}")
    typer.echo(f"Validation MAE: {metrics['validation_mae']:.3f}")
    typer.echo(f"Validation RMSE: {metrics['validation_rmse']:.3f}")
    typer.echo(f"Baseline RMSE: {metrics['baseline_validation_rmse']:.3f}")


def run_p2(
    model_name: str, n_estimators=None, max_depth=None, min_samples_leaf=None,
    stage_of_work="candidate", **_options,
):
    if model_name != "random_forest":
        raise ValueError(f"Modelo no disponible para P2: {model_name!r}")

    datasets = p2.prepare_p2_data()

    train = datasets["train"]
    validation = datasets["validation"]
    periods = datasets["periods"]

    X_validation = validation[p2.FEATURES]
    y_validation = validation[p2.TARGET]

    model = p2.build_p2_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    started = perf_counter()
    model.fit(train[p2.FEATURES], train[p2.TARGET])
    fit_seconds = perf_counter() - started

    started = perf_counter()
    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]
    predict_seconds = perf_counter() - started
    baseline = p2.predict_baseline(validation)

    baseline_metrics = p2.calculate_metrics(y_validation, baseline)

    metrics = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        **p2.calculate_metrics(y_validation, predictions, probabilities),
        **{f"baseline_{name}": value for name, value in baseline_metrics.items()},
        **p2.confusion_counts(y_validation, predictions),
    }

    params = {
        "data_version": get_data_version(),
        "model": model_name,
        **{key: model.get_params()[key] for key in p2.MODEL_PARAMS},
        "n_jobs": model.n_jobs,
        "train_start": str(periods["train"][0].date()),
        "train_end_exclusive": str(periods["train"][1].date()),
        "validation_start": str(periods["validation"][0].date()),
        "validation_end_exclusive": str(periods["validation"][1].date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_context_days": p2.CONTEXT_DAYS,
        "risk_threshold": p2.P2_RISK_THRESHOLD,
        "features": ",".join(p2.FEATURES),
        **datasets["summary"],
    }

    results = p2.build_results(validation, baseline, predictions, probabilities)

    with TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "predicciones_p2.csv"
        results.to_csv(results_path, index=False)

        example = X_validation.head(5)
        run_id, model_uri = log_model_evaluation(
            run_name="p2_random_forest",
            python_model=EstimatorModel(model),
            input_example=example,
            output_example=model.predict(example),
            params=params,
            metrics=metrics,
            tags={"problem": "p2", "run_type": "model", "execution": "cli"},
            artifacts=[(results_path, "data")],
            figures=plot_classification(y_validation, predictions, probabilities),
            training_data=train[p2.FEATURES + [p2.TARGET]],
            target=p2.TARGET,
            stage_of_work=stage_of_work,
        )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Modelo: {model_uri}")
    typer.echo(f"Baseline F1: {metrics['baseline_f1']:.3f}")
    typer.echo(f"F1: {metrics['f1']:.3f}")
    typer.echo(f"ROC-AUC: {metrics['roc_auc']:.3f}")


def run_p3(model_name: str, n_clusters=None, stage_of_work="candidate", **_options):
    if model_name != "kmeans":
        raise ValueError(f"Modelo no disponible para P3: {model_name!r}")

    prepared = p3.prepare_p3_data()
    dataset = prepared["dataset"]
    start, end_exclusive = prepared["period"]

    started = perf_counter()
    scaler, X_scaled = p3.scale_features(dataset)

    scores = p3.evaluate_k_values(X_scaled)
    best_k = n_clusters if n_clusters is not None else p3.select_best_k(scores)
    if best_k not in p3.K_VALUES:
        raise ValueError(f"n_clusters debe pertenecer a {p3.K_VALUES}")

    model = p3.build_p3_model(best_k)
    dataset["cluster"] = model.fit_predict(X_scaled)
    fit_seconds = perf_counter() - started
    model_pipeline = Pipeline([("scaler", scaler), ("kmeans", model)])

    profile, sizes = p3.cluster_profile(dataset)

    params = {
        "data_version": get_data_version(),
        "model": model_name,
        "n_clusters": best_k,
        "random_state": p3.RANDOM_STATE,
        "n_init": P3_N_INIT,
        "k_values": ",".join(map(str, p3.K_VALUES)),
        "train_start": str(start.date()),
        "train_end_exclusive": str(end_exclusive.date()),
        "stations": len(dataset),
        "features": ",".join(p3.FEATURES),
    }

    metrics = {
        "fit_seconds": fit_seconds,
        "silhouette_score": scores[best_k],
        **{f"silhouette_k{k}": score for k, score in scores.items()},
    }

    with TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "clusters_p3.csv"
        dataset.to_csv(results_path, index=False)
        plot_path = plot_station_clusters(dataset, Path(tmp) / "clusters_p3.png")

        example = dataset[p3.FEATURES].head(5)
        run_id, model_uri = log_model_evaluation(
            run_name="p3_kmeans",
            python_model=EstimatorModel(model_pipeline),
            input_example=example,
            output_example=model_pipeline.predict(example),
            params=params,
            metrics=metrics,
            tags={"problem": "p3", "run_type": "model", "execution": "cli"},
            artifacts=[(results_path, "data"), (plot_path, "plots")],
            figures={"cluster_profiles.png": plot_cluster_profiles(dataset, p3.FEATURES)},
            training_data=dataset[p3.FEATURES],
            stage_of_work=stage_of_work,
        )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Modelo: {model_uri}")
    typer.echo(f"Mejor K: {best_k} (silhouette {scores[best_k]:.3f})")
    typer.echo(f"Estaciones por cluster:\n{sizes.to_string()}")
    typer.echo(f"Perfil promedio:\n{profile.to_string()}")


# Cada función recibe el modelo solicitado y sus opciones.
TRAINERS = {
    "p1": run_p1,
    "p2": run_p2,
    "p3": run_p3,
}

@app.command()
def main(
    problem: str = typer.Option("p1", "--problem"),
    model: str = typer.Option(None, "--model", help="Por defecto, el modelo base de cada problema."),
    fit_intercept: str = typer.Option("auto", "--fit-intercept"),
    n_estimators: int | None = typer.Option(None),
    max_depth: int | None = typer.Option(None),
    min_samples_leaf: int | None = typer.Option(None),
    n_clusters: int | None = typer.Option(None),
    alpha: float = typer.Option(-1.0, help="Regularización de Ridge; -1 usa config.py."),
    stage_of_work: str = typer.Option("candidate"),
):
    if problem not in TRAINERS:
        available = ", ".join(TRAINERS)

        raise typer.BadParameter(
            f"Problema no disponible. Opciones: {available}",
            param_hint="--problem",
        )

    if stage_of_work not in {"baseline", "candidate"}:
        raise typer.BadParameter("Usa baseline o candidate", param_hint="--stage-of-work")
    if alpha < 0 and alpha != -1:
        raise typer.BadParameter("alpha debe ser >= 0 o -1 para usar config.py")
    for name, value in {
        "n_estimators": n_estimators, "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf, "n_clusters": n_clusters,
    }.items():
        if value is not None and value < 0:
            raise typer.BadParameter(f"{name} debe ser positivo o 0 para usar config.py")

    if fit_intercept.lower() not in {"auto", "true", "false"}:
        raise typer.BadParameter("Usa true, false o auto", param_hint="--fit-intercept")
    resolved_fit_intercept = (
        P1_FIT_INTERCEPT
        if fit_intercept.lower() == "auto"
        else fit_intercept.lower() == "true"
    )

    requested_model = (model or "auto").strip().strip('"').strip("'")
    if requested_model.lower() in {"", "auto", "none"}:
        requested_model = DEFAULT_MODELS[problem]

    TRAINERS[problem](
        model_name=requested_model,
        fit_intercept=resolved_fit_intercept,
        n_estimators=n_estimators or None,
        max_depth=max_depth or None,
        min_samples_leaf=min_samples_leaf or None,
        n_clusters=n_clusters or None,
        alpha=P1_RIDGE_ALPHA if alpha == -1 else alpha,
        stage_of_work=stage_of_work,
    )


if __name__ == "__main__":
    app()
