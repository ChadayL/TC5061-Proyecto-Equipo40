"""Ejecuta las candidatas de un YAML mediante el mismo MLflow Project."""

import math
from pathlib import Path

import mlflow
import typer
import yaml

from mibici.config import P3_K_VALUES, PROJ_ROOT
from mibici.modeling.tracking import build_run_tags, configure_tracking

app = typer.Typer()


def load_search(path):
    """Valida el plan completo antes de lanzar un entrenamiento."""
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("El plan debe contener question, model y candidates.")
    allowed = {
        "p1": ({"linear_regression", "ridge"}, {"alpha", "fit_intercept"}),
        "p2": ({"random_forest"}, {"n_estimators", "max_depth", "min_samples_leaf"}),
        "p3": ({"kmeans"}, {"n_clusters"}),
    }
    question, model = plan.get("question"), plan.get("model")
    if question not in allowed or model not in allowed[question][0]:
        raise ValueError("Combinación de question y model no disponible.")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates debe ser una lista no vacía.")
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError("Cada candidata debe declarar sus hiperparámetros.")
        if candidate.keys() - allowed[question][1]:
            raise ValueError(f"Parámetros no disponibles para {question}: {candidate}")
        if model == "linear_regression" and "alpha" in candidate:
            raise ValueError("alpha solo se aplica al modelo ridge.")
        for name, value in candidate.items():
            if name == "fit_intercept":
                if not isinstance(value, bool):
                    raise ValueError("fit_intercept debe ser true o false.")
            elif (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name != "alpha" and (not isinstance(value, int) or value == 0))
            ):
                raise ValueError(f"Valor inválido para {name}: {value}")
        if question == "p3" and candidate["n_clusters"] not in P3_K_VALUES:
            raise ValueError(f"n_clusters debe pertenecer a {P3_K_VALUES}")
    if len({tuple(sorted(candidate.items())) for candidate in candidates}) != len(candidates):
        raise ValueError("El plan contiene combinaciones repetidas.")
    return plan


@app.command()
def main(config: Path = typer.Option(..., exists=True, dir_okay=False)):
    plan = load_search(config)
    experiment = configure_tracking()
    question, model = plan["question"], plan["model"]
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"{question}_{model}_search",
    ) as parent:
        mlflow.set_tags(build_run_tags({
            "question": question, "problem": question, "model_family": model,
            "stage_of_work": "search", "run_type": "search",
        }))
        mlflow.log_artifact(str(config), artifact_path="config")
        mlflow.log_param("candidate_count", len(plan["candidates"]))
        for index, candidate in enumerate(plan["candidates"], start=1):
            parameters = {
                "question": question, "model": model, "stage_of_work": "candidate",
                **{key: str(value).lower() if isinstance(value, bool) else value
                   for key, value in candidate.items()},
            }
            # Projects vincula cada ejecución con la corrida activa como padre.
            submitted = mlflow.projects.run(
                str(PROJ_ROOT), entry_point="train", parameters=parameters,
                experiment_id=parent.info.experiment_id, env_manager="local",
                run_name=f"{question}_{model}_{index:02d}",
            )
            mlflow.MlflowClient().log_artifact(
                submitted.run_id, str(config), artifact_path="config",
            )
            typer.echo(f"Candidata {index}: {submitted.run_id}")


if __name__ == "__main__":
    app()
