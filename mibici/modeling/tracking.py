import mlflow

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mibici.config import PROJ_ROOT
from mibici.modeling.p1 import predict_departures


def configure_tracking():
    """Configura el servidor y selecciona el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def log_evaluation(run_name, params, metrics, tags, artifacts=()):
    experiment = configure_tracking()
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
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
):
    experiment = configure_tracking()

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=python_model,
            input_example=input_example,
            signature=mlflow.models.infer_signature(input_example, output_example),
            code_paths=[str(PROJ_ROOT / "mibici")],
        )

        return run.info.run_id, model_info.model_uri


class DepartureModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        return predict_departures(self.pipeline, model_input)
