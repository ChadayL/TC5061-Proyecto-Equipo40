"""Promueve a producción el mejor modelo registrado por pregunta."""

import mlflow
import typer
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mibici.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

app = typer.Typer()
SELECTION = {
    "p1": ("validation_rmse", "ASC"),
    "p2": ("f1", "DESC"),
    "p3": ("silhouette_score", "DESC"),
}


def find_best_logged_run(client, experiment_id, problem):
    metric, direction = SELECTION[problem]
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.problem = '{problem}' and tags.run_type = 'model' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=[f"metrics.{metric} {direction}", "attributes.start_time ASC"],
        max_results=1000,
    )
    # MLflow 3 stores logged models as first-class entities. They do not always
    # appear in client.list_artifacts(run_id), even when the UI shows the model.
    # Build a run_id -> model_id map from the LoggedModel API instead.
    logged_models = client.search_logged_models(
        experiment_ids=[experiment_id],
        max_results=1000,
    )
    model_by_run = {}
    for logged_model in logged_models:
        if logged_model.source_run_id and logged_model.status == "READY":
            model_by_run.setdefault(logged_model.source_run_id, logged_model)

    for run in runs:
        if metric not in run.data.metrics:
            continue
        logged_model = model_by_run.get(run.info.run_id)
        if logged_model is not None:
            return run, metric, logged_model
    raise RuntimeError(
        f"No hay corridas FINISHED de {problem} con métrica {metric} y artefacto model. "
        "Vuelve a ejecutar el entrenamiento actualizado de esa pregunta."
    )


@app.command()
def promote(problem: str = typer.Option(..., "--problem")):
    """Registra la mejor corrida y la marca como versión de producción."""
    if problem not in SELECTION:
        raise typer.BadParameter("Usa p1, p2 o p3", param_hint="--problem")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No existe el experimento {MLFLOW_EXPERIMENT_NAME!r}")

    run, metric, logged_model = find_best_logged_run(
        client, experiment.experiment_id, problem
    )
    registered_name = f"mibici-{problem}"
    try:
        client.get_registered_model(registered_name)
    except MlflowException:
        client.create_registered_model(registered_name)

    # Register directly from the MLflow 3 logged model ID instead of relying on
    # the run-artifact listing implementation.
    model_uri = f"models:/{logged_model.model_id}"
    version = mlflow.register_model(model_uri=model_uri, name=registered_name)
    client.set_registered_model_alias(registered_name, "production", version.version)
    # El alias es la referencia recomendada en MLflow 3; se conserva también la etapa
    # solicitada en la actividad para mostrar Production en la interfaz del Registry.
    client.transition_model_version_stage(
        name=registered_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    source_data = run.data.params.get("data_version", "unknown")
    source_commit = run.data.tags.get("commit", "unknown")
    baseline_key = {"p1": "baseline_validation_rmse", "p2": "baseline_f1"}.get(problem)
    baseline_value = run.data.metrics.get(baseline_key) if baseline_key else None
    selected_value = run.data.metrics[metric]
    if baseline_value is None:
        baseline_result = "not_available"
    else:
        improved = selected_value < baseline_value if problem == "p1" else selected_value > baseline_value
        baseline_result = "better_than_baseline" if improved else "not_better_than_baseline"
    client.set_model_version_tag(registered_name, version.version, "question", problem)
    client.set_model_version_tag(registered_name, version.version, "selection_metric", metric)
    client.set_model_version_tag(
        registered_name, version.version, "selection_value", str(run.data.metrics[metric])
    )
    client.set_model_version_tag(registered_name, version.version, "data_version", source_data)
    client.set_model_version_tag(registered_name, version.version, "commit", source_commit)
    client.set_model_version_tag(registered_name, version.version, "baseline_result", baseline_result)

    typer.echo(f"Modelo: {registered_name}")
    typer.echo(f"Versión: {version.version}")
    typer.echo(f"Corrida fuente: {run.info.run_id}")
    typer.echo(f"{metric}: {run.data.metrics[metric]:.4f}")
    typer.echo("Alias: production | Etapa: Production")


if __name__ == "__main__":
    app()
