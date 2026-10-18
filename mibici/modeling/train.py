from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    log_evaluation,
    log_model_evaluation,
)
from mibici.plots import plot_station_clusters

app = typer.Typer()


def run_p1(model_name: str, fit_intercept: bool):
    if model_name != "linear_regression":
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

    model = build_model_pipeline(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    predictions = predict_departures(model, X_validation)

    metrics = {
        "validation_mae": mean_absolute_error(y_validation, predictions),
        "validation_rmse": mean_squared_error(y_validation, predictions) ** 0.5,
    }

    params = {
        "model": "linear_regression",
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

    wrapped_model = DepartureModel(model)
    example = X_train.head(5)

    run_id, model_uri = log_model_evaluation(
        run_name="p1_linear_regression",
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
    )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Modelo: {model_uri}")
    typer.echo(f"Validation MAE: {metrics['validation_mae']:.3f}")
    typer.echo(f"Validation RMSE: {metrics['validation_rmse']:.3f}")


def run_p2(model_name: str, **_options):
    if model_name != "random_forest":
        raise ValueError(f"Modelo no disponible para P2: {model_name!r}")

    datasets = p2.prepare_p2_data()

    train = datasets["train"]
    validation = datasets["validation"]
    periods = datasets["periods"]

    X_validation = validation[p2.FEATURES]
    y_validation = validation[p2.TARGET]

    model = p2.build_p2_model()
    model.fit(train[p2.FEATURES], train[p2.TARGET])

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]
    baseline = p2.predict_baseline(validation)

    baseline_metrics = p2.calculate_metrics(y_validation, baseline)

    metrics = {
        **p2.calculate_metrics(y_validation, predictions, probabilities),
        **{f"baseline_{name}": value for name, value in baseline_metrics.items()},
        **p2.confusion_counts(y_validation, predictions),
    }

    params = {
        "model": model_name,
        **p2.MODEL_PARAMS,
        "train_start": str(periods["train"][0].date()),
        "train_end_exclusive": str(periods["train"][1].date()),
        "validation_start": str(periods["validation"][0].date()),
        "validation_end_exclusive": str(periods["validation"][1].date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_context_days": p2.CONTEXT_DAYS,
        "features": ",".join(p2.FEATURES),
        **datasets["summary"],
    }

    results = p2.build_results(validation, baseline, predictions, probabilities)

    with TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "predicciones_p2.csv"
        results.to_csv(results_path, index=False)

        run_id = log_evaluation(
            run_name="p2_random_forest",
            params=params,
            metrics=metrics,
            tags={"problem": "p2", "run_type": "model", "execution": "cli"},
            artifacts=[results_path],
        )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Baseline F1: {metrics['baseline_f1']:.3f}")
    typer.echo(f"F1: {metrics['f1']:.3f}")
    typer.echo(f"ROC-AUC: {metrics['roc_auc']:.3f}")


def run_p3(model_name: str, **_options):
    if model_name != "kmeans":
        raise ValueError(f"Modelo no disponible para P3: {model_name!r}")

    prepared = p3.prepare_p3_data()
    dataset = prepared["dataset"]
    start, end_exclusive = prepared["period"]

    _, X_scaled = p3.scale_features(dataset)

    scores = p3.evaluate_k_values(X_scaled)
    best_k = p3.select_best_k(scores)

    model = p3.build_p3_model(best_k)
    dataset["cluster"] = model.fit_predict(X_scaled)

    profile, sizes = p3.cluster_profile(dataset)

    params = {
        "model": model_name,
        "n_clusters": best_k,
        "random_state": p3.RANDOM_STATE,
        "k_values": ",".join(map(str, p3.K_VALUES)),
        "train_start": str(start.date()),
        "train_end_exclusive": str(end_exclusive.date()),
        "stations": len(dataset),
        "features": ",".join(p3.FEATURES),
    }

    metrics = {
        "silhouette_score": scores[best_k],
        **{f"silhouette_k{k}": score for k, score in scores.items()},
    }

    with TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "clusters_p3.csv"
        dataset.to_csv(results_path, index=False)
        plot_path = plot_station_clusters(dataset, Path(tmp) / "clusters_p3.png")

        run_id = log_evaluation(
            run_name="p3_kmeans",
            params=params,
            metrics=metrics,
            tags={"problem": "p3", "run_type": "model", "execution": "cli"},
            artifacts=[results_path, plot_path],
        )

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Mejor K: {best_k} (silhouette {scores[best_k]:.3f})")
    typer.echo(f"Estaciones por cluster:\n{sizes.to_string()}")
    typer.echo(f"Perfil promedio:\n{profile.to_string()}")


# Cada función recibe el modelo solicitado y sus opciones.
TRAINERS = {
    "p1": run_p1,
    "p2": run_p2,
    "p3": run_p3,
}

DEFAULT_MODELS = {
    "p1": "linear_regression",
    "p2": "random_forest",
    "p3": "kmeans",
}


@app.command()
def main(
    problem: str = typer.Option("p1", "--problem"),
    model: str = typer.Option(None, "--model", help="Por defecto, el modelo base de cada problema."),
    fit_intercept: bool = typer.Option(
        True,
        "--fit-intercept/--no-fit-intercept",
    ),
):
    if problem not in TRAINERS:
        available = ", ".join(TRAINERS)

        raise typer.BadParameter(
            f"Problema no disponible. Opciones: {available}",
            param_hint="--problem",
        )

    TRAINERS[problem](
        model_name=model or DEFAULT_MODELS[problem],
        fit_intercept=fit_intercept,
    )


if __name__ == "__main__":
    app()
