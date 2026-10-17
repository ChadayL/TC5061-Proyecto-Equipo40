import mlflow
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
)

from mibici.dataset import load_raw
from mibici.features import (
    build_p1_features,
    build_p2_features,
    build_p3_features,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

TRAIN_END = "2026-09-01"


# ============================================================
# CONFIGURACIÓN P1
# ============================================================

FEATURES_P1 = [
    "Origen_Id",
    "dia_semana",
    "fin_semana",
    "lag_1",
    "lag_7",
    "promedio_7d",
    "latitude",
    "longitude",
    "capacity",
]

TARGET_P1 = "viajes"


# ============================================================
# CONFIGURACIÓN P2
# ============================================================

FEATURES_P2 = [
    "salidas",
    "llegadas",
    "balance",
    "capacity",
    "presion_capacidad",
    "balance_lag_1",
    "balance_lag_7",
    "presion_lag_1",
    "presion_promedio_7d",
    "dia_semana",
    "fin_semana",
]

TARGET_P2 = "riesgo_manana"


# ============================================================
# CONFIGURACIÓN P3
# ============================================================

FEATURES_P3 = [
    "salidas_totales",
    "llegadas_totales",
    "salidas_promedio",
    "salidas_std",
    "proporcion_fin_semana",
    "balance",
    "ratio_llegadas_salidas",
    "capacity",
]

K_VALUES_P3 = [2, 3, 4, 5, 6]

RANDOM_STATE_P3 = 42


# ============================================================
# MÉTRICAS P1
# ============================================================

def calculate_metrics_p1(y_true, y_pred):
    """
    Calcula métricas de regresión para P1.
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
# MÉTRICAS P2
# ============================================================

def calculate_metrics_p2(y_true, y_pred, y_prob):
    """
    Calcula métricas de clasificación para P2.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ROC-AUC necesita que existan ambas clases en y_true.
    if y_true.nunique() > 1:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
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

    X_train = train[FEATURES_P1]
    y_train = train[TARGET_P1]

    X_test = test[FEATURES_P1]
    y_test = test[TARGET_P1]

    # --------------------------------------------------------
    # 4. BASELINE
    # --------------------------------------------------------

    print("\n4. Evaluando baseline...")

    baseline_pred = test["lag_7"]

    baseline_metrics = calculate_metrics_p1(
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

    model_metrics = calculate_metrics_p1(
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

    if baseline_metrics["MAE"] != 0:
        improvement = (
            (
                baseline_metrics["MAE"]
                - model_metrics["MAE"]
            )
            / baseline_metrics["MAE"]
            * 100
        )
    else:
        improvement = 0.0

    print("\n==============================")
    print("COMPARACIÓN FINAL P1")
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
    # 9. TABLA DE RESULTADOS
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
    # 11. MLFLOW
    # --------------------------------------------------------

    print("7. Registrando experimento en MLflow...")

    mlflow.set_experiment(
        "mibici-equipo40"
    )

    with mlflow.start_run(
        run_name="P1-random-forest"
    ):

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

        mlflow.log_metric(
            "mae_improvement_percent",
            improvement,
        )

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

        mlflow.log_artifact(
            artifact_file,
            artifact_path="resultados",
        )

    print(
        "   Experimento P1 registrado correctamente."
    )

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
# ENTRENAMIENTO P2
# ============================================================

def train_p2():
    """
    Entrena y evalúa el modelo de clasificación para P2.

    P2:
    ¿La estación tendrá riesgo de saturación
    durante el día siguiente?
    """

    print("\n==============================")
    print("P2 - PREDICCIÓN DE RIESGO")
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

    print("2. Construyendo features P2...")

    dataset = build_p2_features(
        viajes,
        estaciones,
    )

    print(
        f"   Observaciones disponibles: {len(dataset)}"
    )

    print(
        f"   Casos sin riesgo: "
        f"{(dataset[TARGET_P2] == 0).sum()}"
    )

    print(
        f"   Casos con riesgo: "
        f"{(dataset[TARGET_P2] == 1).sum()}"
    )

    print(
        f"   Porcentaje de riesgo: "
        f"{dataset[TARGET_P2].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 3. PARTICIÓN TEMPORAL
    # --------------------------------------------------------

    print("\n3. Realizando partición temporal...")

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

    print(
        f"   Riesgo en train: "
        f"{train[TARGET_P2].mean() * 100:.2f}%"
    )

    print(
        f"   Riesgo en test: "
        f"{test[TARGET_P2].mean() * 100:.2f}%"
    )

    X_train = train[FEATURES_P2]
    y_train = train[TARGET_P2]

    X_test = test[FEATURES_P2]
    y_test = test[TARGET_P2]

    # --------------------------------------------------------
    # 4. BASELINE
    # --------------------------------------------------------

    print("\n4. Evaluando baseline...")

    # Baseline:
    # asumimos que el riesgo de mañana será igual
    # al riesgo observado actualmente.

    baseline_pred = test["riesgo_actual"].astype(int)

    baseline_metrics = {
        "accuracy": accuracy_score(
            y_test,
            baseline_pred,
        ),
        "precision": precision_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            baseline_pred,
            zero_division=0,
        ),
    }

    print(
        f"   Accuracy:  "
        f"{baseline_metrics['accuracy']:.3f}"
    )

    print(
        f"   Precision: "
        f"{baseline_metrics['precision']:.3f}"
    )

    print(
        f"   Recall:    "
        f"{baseline_metrics['recall']:.3f}"
    )

    print(
        f"   F1:        "
        f"{baseline_metrics['f1']:.3f}"
    )

    # --------------------------------------------------------
    # 5. RANDOM FOREST CLASSIFIER
    # --------------------------------------------------------

    print("\n5. Entrenando Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
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

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # 7. MÉTRICAS
    # --------------------------------------------------------

    model_metrics = calculate_metrics_p2(
        y_test,
        predictions,
        probabilities,
    )

    print("\n6. Resultados Random Forest:")

    print(
        f"   Accuracy:  "
        f"{model_metrics['accuracy']:.3f}"
    )

    print(
        f"   Precision: "
        f"{model_metrics['precision']:.3f}"
    )

    print(
        f"   Recall:    "
        f"{model_metrics['recall']:.3f}"
    )

    print(
        f"   F1:        "
        f"{model_metrics['f1']:.3f}"
    )

    print(
        f"   ROC-AUC:   "
        f"{model_metrics['roc_auc']:.3f}"
    )

    # --------------------------------------------------------
    # 8. MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n7. Matriz de confusión:")

    print(
        f"   Verdaderos negativos: {tn}"
    )

    print(
        f"   Falsos positivos:     {fp}"
    )

    print(
        f"   Falsos negativos:     {fn}"
    )

    print(
        f"   Verdaderos positivos: {tp}"
    )

    # --------------------------------------------------------
    # 9. COMPARACIÓN CONTRA BASELINE
    # --------------------------------------------------------

    print("\n==============================")
    print("COMPARACIÓN FINAL P2")
    print("==============================")

    print(
        f"Baseline F1:      "
        f"{baseline_metrics['f1']:.3f}"
    )

    print(
        f"Random Forest F1: "
        f"{model_metrics['f1']:.3f}"
    )

    print(
        f"Random Forest ROC-AUC: "
        f"{model_metrics['roc_auc']:.3f}"
    )

    print("==============================\n")

    # --------------------------------------------------------
    # 10. TABLA DE RESULTADOS
    # --------------------------------------------------------

    results = test[
        [
            "fecha",
            "estacion_id",
            "salidas",
            "llegadas",
            "balance",
            "capacity",
            "presion_capacidad",
            "riesgo_actual",
            "riesgo_manana",
        ]
    ].copy()

    results["baseline"] = baseline_pred.values

    results["prediccion_rf"] = predictions

    results["probabilidad_riesgo"] = probabilities

    # --------------------------------------------------------
    # 11. GUARDAR CSV
    # --------------------------------------------------------

    artifact_file = "predicciones_p2.csv"

    results.to_csv(
        artifact_file,
        index=False,
    )

    print(
        f"8. Resultados guardados en {artifact_file}"
    )

    # --------------------------------------------------------
    # 12. MLFLOW
    # --------------------------------------------------------

    print("9. Registrando experimento en MLflow...")

    mlflow.set_experiment(
        "mibici-equipo40"
    )

    with mlflow.start_run(
        run_name="P2-random-forest"
    ):

        # Parámetros generales

        mlflow.log_param(
            "pregunta",
            "P2",
        )

        mlflow.log_param(
            "modelo",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "train_end",
            TRAIN_END,
        )

        # Parámetros del modelo

        mlflow.log_param(
            "n_estimators",
            200,
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
            "class_weight",
            "balanced",
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # Dataset

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

        # Métricas del modelo

        mlflow.log_metric(
            "accuracy",
            model_metrics["accuracy"],
        )

        mlflow.log_metric(
            "precision",
            model_metrics["precision"],
        )

        mlflow.log_metric(
            "recall",
            model_metrics["recall"],
        )

        mlflow.log_metric(
            "f1",
            model_metrics["f1"],
        )

        mlflow.log_metric(
            "roc_auc",
            model_metrics["roc_auc"],
        )

        # Baseline

        mlflow.log_metric(
            "baseline_accuracy",
            baseline_metrics["accuracy"],
        )

        mlflow.log_metric(
            "baseline_precision",
            baseline_metrics["precision"],
        )

        mlflow.log_metric(
            "baseline_recall",
            baseline_metrics["recall"],
        )

        mlflow.log_metric(
            "baseline_f1",
            baseline_metrics["f1"],
        )

        # Matriz de confusión

        mlflow.log_metric(
            "true_negatives",
            int(tn),
        )

        mlflow.log_metric(
            "false_positives",
            int(fp),
        )

        mlflow.log_metric(
            "false_negatives",
            int(fn),
        )

        mlflow.log_metric(
            "true_positives",
            int(tp),
        )

        # Tags

        mlflow.set_tag(
            "equipo",
            "40",
        )

        mlflow.set_tag(
            "pregunta",
            "P2",
        )

        mlflow.set_tag(
            "tipo_tarea",
            "clasificacion",
        )

        mlflow.set_tag(
            "modelo",
            "RandomForest",
        )

        # Artefacto

        mlflow.log_artifact(
            artifact_file,
            artifact_path="resultados",
        )

    print(
        "   Experimento P2 registrado correctamente."
    )

    print("\n==============================")
    print("P2 COMPLETADO CORRECTAMENTE")
    print("==============================\n")

    return {
        "model": model,
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "confusion_matrix": cm,
        "results": results,
    }


# ============================================================
# ENTRENAMIENTO P3
# ============================================================

def train_p3():
    """
    Entrena y evalúa el modelo de clustering para P3.

    P3:
    ¿Qué grupos de estaciones comparten patrones de uso
    y dónde se ubican en la ciudad?

    Se utiliza K-Means sobre variables de comportamiento
    de las estaciones. Las coordenadas geográficas se
    conservan únicamente para visualizar los clusters.
    """

    print("\n==============================")
    print("P3 - CLUSTERING DE ESTACIONES")
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

    print("2. Construyendo features P3...")

    dataset = build_p3_features(
        viajes,
        estaciones,
    )

    print(
        f"   Estaciones disponibles: {len(dataset)}"
    )

    # --------------------------------------------------------
    # 3. PREPARAR VARIABLES
    # --------------------------------------------------------

    print("\n3. Preparando variables...")

    X = dataset[FEATURES_P3].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # 4. EVALUAR DIFERENTES VALORES DE K
    # --------------------------------------------------------

    print("\n4. Evaluando número de clusters...")

    resultados_k = {}

    for k in K_VALUES_P3:

        modelo_k = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE_P3,
            n_init=10,
        )

        labels_k = modelo_k.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels_k,
        )

        resultados_k[k] = score

        print(
            f"   K={k} | "
            f"Silhouette Score: {score:.3f}"
        )

    # --------------------------------------------------------
    # 5. SELECCIONAR MEJOR K
    # --------------------------------------------------------

    mejor_k = max(
        resultados_k,
        key=resultados_k.get,
    )

    mejor_silhouette = resultados_k[
        mejor_k
    ]

    print("\n==============================")
    print("SELECCIÓN DE CLUSTERS")
    print("==============================")

    print(
        f"Mejor K: {mejor_k}"
    )

    print(
        f"Silhouette Score: "
        f"{mejor_silhouette:.3f}"
    )

    print("==============================\n")

    # --------------------------------------------------------
    # 6. ENTRENAR MODELO FINAL
    # --------------------------------------------------------

    print("5. Entrenando modelo K-Means final...")

    model = KMeans(
        n_clusters=mejor_k,
        random_state=RANDOM_STATE_P3,
        n_init=10,
    )

    clusters = model.fit_predict(
        X_scaled
    )

    dataset["cluster"] = clusters

    # --------------------------------------------------------
    # 7. PERFIL DE LOS CLUSTERS
    # --------------------------------------------------------

    print("\n6. Perfil promedio de clusters:")

    profile = (
        dataset
        .groupby("cluster")[FEATURES_P3]
        .mean()
        .round(2)
    )

    print(profile)

    print("\n7. Número de estaciones por cluster:")

    cluster_sizes = (
        dataset["cluster"]
        .value_counts()
        .sort_index()
    )

    print(cluster_sizes)

    # --------------------------------------------------------
    # 8. GUARDAR RESULTADOS
    # --------------------------------------------------------

    results_file = "clusters_p3.csv"

    dataset.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\n8. Resultados guardados en {results_file}"
    )

    # --------------------------------------------------------
    # 9. GRÁFICA GEOGRÁFICA
    # --------------------------------------------------------

    print(
        "9. Generando gráfica de clusters..."
    )

    artifact_file = "clusters_p3.png"

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        dataset["longitude"],
        dataset["latitude"],
        c=dataset["cluster"],
        s=40,
        alpha=0.75,
    )

    plt.xlabel(
        "Longitud"
    )

    plt.ylabel(
        "Latitud"
    )

    plt.title(
        "P3 - Clusters de estaciones MiBici"
    )

    plt.colorbar(
        scatter,
        label="Cluster",
    )

    plt.tight_layout()

    plt.savefig(
        artifact_file,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # 10. MLFLOW
    # --------------------------------------------------------

    print(
        "10. Registrando experimento en MLflow..."
    )

    mlflow.set_experiment(
        "mibici-equipo40"
    )

    with mlflow.start_run(
        run_name="P3-kmeans"
    ):

        # Parámetros generales

        mlflow.log_param(
            "pregunta",
            "P3",
        )

        mlflow.log_param(
            "modelo",
            "KMeans",
        )

        mlflow.log_param(
            "n_clusters",
            mejor_k,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE_P3,
        )

        mlflow.log_param(
            "n_estaciones",
            len(dataset),
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES_P3),
        )

        # Métrica principal

        mlflow.log_metric(
            "silhouette_score",
            mejor_silhouette,
        )

        # Silhouette para cada K evaluado

        for k, score in resultados_k.items():

            mlflow.log_metric(
                f"silhouette_k{k}",
                score,
            )

        # Tags

        mlflow.set_tag(
            "equipo",
            "40",
        )

        mlflow.set_tag(
            "pregunta",
            "P3",
        )

        mlflow.set_tag(
            "tipo_tarea",
            "clustering",
        )

        mlflow.set_tag(
            "modelo",
            "KMeans",
        )

        # Artefactos

        mlflow.log_artifact(
            artifact_file,
            artifact_path="resultados",
        )

        mlflow.log_artifact(
            results_file,
            artifact_path="resultados",
        )

    print(
        "   Experimento P3 registrado correctamente."
    )

    print("\n==============================")
    print("P3 COMPLETADO CORRECTAMENTE")
    print("==============================\n")

    return {
        "model": model,
        "scaler": scaler,
        "best_k": mejor_k,
        "silhouette_score": mejor_silhouette,
        "scores": resultados_k,
        "profile": profile,
        "results": dataset,
    }


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    train_p1()

    train_p2()

    train_p3()