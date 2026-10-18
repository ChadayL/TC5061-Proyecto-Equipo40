"""P2: riesgo de desbalance por estación para el día siguiente.

Lógica trasladada desde el train.py del equipo. La carga usa load_stage("common")
y las particiones de get_split_periods().
Pendiente de revisión: fórmula y umbral de riesgo (features.build_p2_features)
e inclusión de estaciones sin actividad.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mibici.dataset import get_split_periods
from mibici.features import build_p2_features
from mibici.pipeline import load_stage

FEATURES = [
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

TARGET = "riesgo_manana"

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}

RESULT_COLUMNS = [
    "fecha",
    "fecha_objetivo",
    "estacion_id",
    "salidas",
    "llegadas",
    "balance",
    "capacity",
    "presion_capacidad",
    "riesgo_actual",
    "riesgo_manana",
]


# balance_lag_7 y presion_promedio_7d miran hasta 7 días antes del día evaluado.
CONTEXT_DAYS = 7


def build_period_dataset(trips, stations, start, end_exclusive):
    """Construye P2 y conserva las filas cuya fecha objetivo pertenece al periodo.

    Cada fila predice al cierre de `fecha` el riesgo de `fecha_objetivo` (= fecha + 1).
    """
    dataset = build_p2_features(trips, stations)
    dataset["fecha_objetivo"] = dataset["fecha"] + pd.Timedelta(days=1)

    in_period = dataset["fecha_objetivo"].ge(start) & dataset["fecha_objetivo"].lt(end_exclusive)

    return dataset.loc[in_period].copy()


def summarize_dates(dataset, name):
    """Fechas de predicción y objetivo realmente evaluadas."""
    return {
        f"{name}_first_prediction_date": str(dataset["fecha"].min().date()),
        f"{name}_last_prediction_date": str(dataset["fecha"].max().date()),
        f"{name}_first_target_date": str(dataset["fecha_objetivo"].min().date()),
        f"{name}_last_target_date": str(dataset["fecha_objetivo"].max().date()),
        f"{name}_target_days": int(dataset["fecha_objetivo"].nunique()),
    }


def prepare_p2_data():
    """Construye train y validation de P2 asignando cada fila por su fecha objetivo."""
    periods = get_split_periods()
    validation_start, validation_end = periods["validation"]

    train_data = load_stage("common", split="train")
    validation_data = load_stage("common", split="validation")

    train = build_period_dataset(
        train_data["viajes"], train_data["estaciones"], *periods["train"]
    )

    # La primera fila de validation predice validation_start al cierre del día anterior;
    # sus rezagos llegan hasta CONTEXT_DAYS días antes de esa fecha de predicción.
    first_needed_day = validation_start - pd.Timedelta(days=1 + CONTEXT_DAYS)

    # Contexto por fecha de fin: incluye toda llegada desde first_needed_day,
    # sin suponer una duración máxima de los viajes.
    train_trips = train_data["viajes"]
    context = train_trips.loc[train_trips["Fin_del_viaje"].ge(first_needed_day)]

    validation = build_period_dataset(
        pd.concat([context, validation_data["viajes"]], ignore_index=True),
        validation_data["estaciones"],
        validation_start,
        validation_end,
    )

    return {
        "train": train,
        "validation": validation,
        "periods": periods,
        "summary": {
            "validation_context_trips": len(context),
            **summarize_dates(train, "train"),
            **summarize_dates(validation, "validation"),
        },
    }


def build_p2_model():
    return RandomForestClassifier(**MODEL_PARAMS, n_jobs=-1)


def predict_baseline(data):
    """Baseline: el riesgo de mañana es igual al riesgo observado hoy."""
    return data["riesgo_actual"].astype(int)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Métricas de clasificación; ROC-AUC solo si se reciben probabilidades."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        # ROC-AUC necesita que existan ambas clases en y_true.
        metrics["roc_auc"] = (
            roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else 0.0
        )

    return metrics


def confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def build_results(test, baseline, predictions, probabilities):
    results = test[RESULT_COLUMNS].copy()
    results["baseline"] = pd.Series(baseline).to_numpy()
    results["prediccion_rf"] = predictions
    results["probabilidad_riesgo"] = probabilities
    return results
