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
    "estacion_id",
    "salidas",
    "llegadas",
    "balance",
    "capacity",
    "presion_capacidad",
    "riesgo_actual",
    "riesgo_manana",
]


def build_split_dataset(split, start, end_exclusive):
    """Construye P2 con los viajes de una partición."""
    data = load_stage("common", split=split)
    dataset = build_p2_features(data["viajes"], data["estaciones"])

    # Solo días cuyo objetivo (el día siguiente) también pertenece a la partición.
    # Excluye además el día extra que generan llegadas posteriores al corte.
    last_day = end_exclusive - pd.Timedelta(days=1)

    return dataset.loc[dataset["fecha"].ge(start) & dataset["fecha"].lt(last_day)].copy()


def prepare_p2_data():
    """Construye train y validation de P2 por separado."""
    periods = get_split_periods()

    return {
        "train": build_split_dataset("train", *periods["train"]),
        "validation": build_split_dataset("validation", *periods["validation"]),
        "periods": periods,
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
