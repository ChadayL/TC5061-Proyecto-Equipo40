import subprocess

import mlflow

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
)
from mibici.modeling.p1 import predict_departures

TEAM_NAME = "Equipo 40"

def build_run_tags(tags):
    """Añade a las etiquetas del run el equipo y el commit del código."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # MLflow Projects en Windows puede no heredar el PATH de Git.
        commit = "unknown"

    return {**tags, "team": TEAM_NAME, "commit": commit}

def configure_tracking():
    """Configura el servidor y selecciona el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def log_effective_params(params, run):
    """Registra valores efectivos sin cambiar los del MLflow Project."""
    existing = run.data.params
    to_log = {}
    for key, value in params.items():
        if key in existing and existing[key] != str(value):
            to_log[f"effective_{key}"] = value
        elif key not in existing:
            to_log[key] = value
    if to_log:
        mlflow.log_params(to_log)


def log_evaluation(run_name, params, metrics, tags, artifacts=()):
    experiment = configure_tracking()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=build_run_tags(tags),
    ) as run:
        log_effective_params(params, run)
        mlflow.log_metrics(metrics)
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="resultados")
        return run.info.run_id
    

def log_model_evaluation(
    run_name,
    python_model,
    input_example,
    output_example,
    params,
    metrics,
    tags,
    artifacts=(),
):
    experiment = configure_tracking()

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=build_run_tags(tags),
    ) as run:
        log_effective_params(params, run)
        mlflow.log_metrics(metrics)
        for path in artifacts:
            mlflow.log_artifact(str(path), artifact_path="resultados")

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=python_model,
            input_example=input_example,
            signature=mlflow.models.infer_signature(input_example, output_example),
            code_paths=[str(PROJ_ROOT / "mibici")],
        )

        return run.info.run_id, model_info.model_uri


class EstimatorModel(mlflow.pyfunc.PythonModel):
    """Adapter pyfunc para modelos sklearn de clasificación y clustering."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, context, model_input, params=None):
        return self.estimator.predict(model_input)


class DepartureModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        return predict_departures(self.pipeline, model_input)
