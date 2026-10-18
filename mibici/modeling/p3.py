"""P3: grupos de estaciones con patrones de uso similares.

Lógica trasladada desde el train.py del equipo. Usa solo la partición train
cargada con load_stage("common").
Pendiente de revisión: totales sin normalizar por días activos y uso del
catálogo (features.build_p3_features).
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mibici.data_preparation.scaling import build_scaler
from mibici.dataset import get_split_periods
from mibici.features import build_p3_features
from mibici.pipeline import load_stage

FEATURES = [
    "salidas_totales",
    "llegadas_totales",
    "salidas_promedio",
    "salidas_std",
    "proporcion_fin_semana",
    "balance",
    "ratio_llegadas_salidas",
    "capacity",
]

K_VALUES = [2, 3, 4, 5, 6]

RANDOM_STATE = 42


def prepare_p3_data():
    """Construye una fila por estación con sus patrones de uso en train."""
    data = load_stage("common", split="train")

    return {
        "dataset": build_p3_features(data["viajes"], data["estaciones"]),
        "period": get_split_periods()["train"],
    }


def scale_features(dataset):
    scaler = build_scaler()
    return scaler, scaler.fit_transform(dataset[FEATURES])


def build_p3_model(n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)


def evaluate_k_values(X_scaled, k_values=K_VALUES):
    """Silhouette de cada número de clusters evaluado."""
    return {
        k: silhouette_score(X_scaled, build_p3_model(k).fit_predict(X_scaled))
        for k in k_values
    }


def select_best_k(scores):
    return max(scores, key=scores.get)


def cluster_profile(dataset):
    """Promedio de cada variable y número de estaciones por cluster."""
    profile = dataset.groupby("cluster")[FEATURES].mean().round(2)
    sizes = dataset["cluster"].value_counts().sort_index()
    return profile, sizes
