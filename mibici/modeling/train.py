import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mibici.dataset import load_raw
from mibici.features import build_p1_features


# ============================================================
# CONFIGURACIÓN P1
# ============================================================

TRAIN_END = "2026-09-01"

FEATURES = [
    "Origen_Id",
    "dia_semana",
    "fin_semana",
    "lag_1",
    "lag_7",
    "promedio_7d",
    "latitude",
    "longitude",
    "dpcapacity",
]

TARGET = "viajes"


# ============================================================
# FUNCIÓN DE MÉTRICAS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calcula las métricas utilizadas para evaluar P1.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# ENTRENAMIENTO P1
# ============================================================

def train_p1():
    """
    Entrena y evalúa el modelo de regresión para P1.

    P1:
    ¿Cuántos viajes se iniciarán en una estación
    durante el día siguiente?

    Train:
    Agosto 2026.

    Test:
    Septiembre 2026.
    """

    print("\n==============================")
    print("P1 - PREDICCIÓN DE DEMANDA")
    print("==============================\n")

    # --------------------------------------------------------
    # 1. CARGAR DATOS
    # --------------------------------------------------------

    print("1. Cargando datos...")

    datos = load_raw()

    viajes = datos["viajes"]
    estaciones = datos["estaciones"]

    # --------------------------------------------------------
    # 2. CONSTRUIR FEATURES
    # --------------------------------------------------------

    print("2. Construyendo features...")

    dataset = build_p1_features(
        viajes,
        estaciones,
    )

    print(
        f"   Observaciones disponibles: {len(dataset)}"
    )

    # --------------------------------------------------------
    # 3. PARTICIÓN TEMPORAL
    # --------------------------------------------------------

    print("3. Realizando partición temporal...")

    train = dataset[
        dataset["fecha"] < TRAIN_END
    ].copy()

    test = dataset[
        dataset["fecha"] >= TRAIN_END
    ].copy()

    print(
        f"   Train: {len(train)} observaciones"
    )

    print(
        f"   Test:  {len(test)} observaciones"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # --------------------------------------------------------
    # 4. BASELINE
    # --------------------------------------------------------

    print("\n4. Evaluando baseline...")

    baseline_pred = test["lag_7"]

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_pred,
    )

    print(
        f"   MAE:  {baseline_metrics['MAE']:.2f}"
    )

    print(
        f"   RMSE: {baseline_metrics['RMSE']:.2f}"
    )

    print(
        f"   R2:   {baseline_metrics['R2']:.3f}"
    )

    # --------------------------------------------------------
    # 5. RANDOM FOREST
    # --------------------------------------------------------

    print("\n5. Entrenando Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 6. PREDICCIONES
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # 7. EVALUACIÓN
    # --------------------------------------------------------

    model_metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("\n6. Resultados Random Forest:")

    print(
        f"   MAE:  {model_metrics['MAE']:.2f}"
    )

    print(
        f"   RMSE: {model_metrics['RMSE']:.2f}"
    )

    print(
        f"   R2:   {model_metrics['R2']:.3f}"
    )

    # --------------------------------------------------------
    # 8. COMPARACIÓN CONTRA BASELINE
    # --------------------------------------------------------

    improvement = (
        (
            baseline_metrics["MAE"]
            - model_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
        * 100
    )

    print("\n==============================")
    print("COMPARACIÓN FINAL")
    print("==============================")

    print(
        f"Baseline MAE:      "
        f"{baseline_metrics['MAE']:.2f}"
    )

    print(
        f"Random Forest MAE: "
        f"{model_metrics['MAE']:.2f}"
    )

    print(
        f"Mejora vs baseline: "
        f"{improvement:.1f}%"
    )

    print("==============================\n")

    # --------------------------------------------------------
    # 9. CREAR TABLA DE RESULTADOS
    # --------------------------------------------------------

    results = test[
        [
            "fecha",
            "Origen_Id",
            "viajes",
        ]
    ].copy()

    results["baseline"] = baseline_pred.values
    results["prediccion_rf"] = predictions

    # --------------------------------------------------------
    # 10. GUARDAR ARTEFACTO
    # --------------------------------------------------------

    artifact_file = "predicciones_p1.csv"

    results.to_csv(
        artifact_file,
        index=False,
    )

    # --------------------------------------------------------
    # 11. REGISTRAR EXPERIMENTO EN MLFLOW
    # --------------------------------------------------------

    print("7. Registrando experimento en MLflow...")

    mlflow.set_experiment(
        "mibici-equipo40"
    )

    with mlflow.start_run(
        run_name="P1-random-forest"
    ):

        # ----------------------------------------------------
        # PARÁMETROS GENERALES
        # ----------------------------------------------------

        mlflow.log_param(
            "pregunta",
            "P1",
        )

        mlflow.log_param(
            "modelo",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "train_end",
            TRAIN_END,
        )

        # ----------------------------------------------------
        # PARÁMETROS DEL RANDOM FOREST
        # ----------------------------------------------------

        mlflow.log_param(
            "n_estimators",
            100,
        )

        mlflow.log_param(
            "max_depth",
            15,
        )

        mlflow.log_param(
            "min_samples_leaf",
            2,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # ----------------------------------------------------
        # INFORMACIÓN DEL DATASET
        # ----------------------------------------------------

        mlflow.log_param(
            "train_observations",
            len(train),
        )

        mlflow.log_param(
            "test_observations",
            len(test),
        )

        mlflow.log_param(
            "total_observations",
            len(dataset),
        )

        # ----------------------------------------------------
        # MÉTRICAS RANDOM FOREST
        # ----------------------------------------------------

        mlflow.log_metric(
            "mae",
            model_metrics["MAE"],
        )

        mlflow.log_metric(
            "rmse",
            model_metrics["RMSE"],
        )

        mlflow.log_metric(
            "r2",
            model_metrics["R2"],
        )

        # ----------------------------------------------------
        # MÉTRICAS BASELINE
        # ----------------------------------------------------

        mlflow.log_metric(
            "baseline_mae",
            baseline_metrics["MAE"],
        )

        mlflow.log_metric(
            "baseline_rmse",
            baseline_metrics["RMSE"],
        )

        mlflow.log_metric(
            "baseline_r2",
            baseline_metrics["R2"],
        )

        # ----------------------------------------------------
        # MEJORA CONTRA BASELINE
        # ----------------------------------------------------

        mlflow.log_metric(
            "mae_improvement_percent",
            improvement,
        )

        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        mlflow.set_tag(
            "equipo",
            "40",
        )

        mlflow.set_tag(
            "pregunta",
            "P1",
        )

        mlflow.set_tag(
            "tipo_tarea",
            "regresion",
        )

        mlflow.set_tag(
            "modelo",
            "RandomForest",
        )

        # ----------------------------------------------------
        # ARTEFACTO
        # ----------------------------------------------------

        mlflow.log_artifact(
            artifact_file,
            artifact_path="resultados",
        )

    print(
        "   Experimento registrado correctamente."
    )

    # --------------------------------------------------------
    # 12. FINAL
    # --------------------------------------------------------

    print("\n==============================")
    print("P1 COMPLETADO CORRECTAMENTE")
    print("==============================\n")

    return {
        "model": model,
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "results": results,
    }


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    train_p1()