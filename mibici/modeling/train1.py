import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    log_model_evaluation,
)

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


# Cada función recibe el modelo solicitado y sus opciones.
TRAINERS = {
    "p1": run_p1,
}


@app.command()
def main(
    problem: str = typer.Option("p1", "--problem"),
    model: str = typer.Option("linear_regression", "--model"),
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
        model_name=model,
        fit_intercept=fit_intercept,
    )


if __name__ == "__main__":
    app()
