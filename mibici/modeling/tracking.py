import mlflow

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def configure_tracking():
    """Configura el servidor y selecciona el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )
    
    
def log_evaluation(run_name, params, metrics, tags):
    """Registra la evaluación y devuelve su identificador."""
    experiment = configure_tracking()

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        return run.info.run_id