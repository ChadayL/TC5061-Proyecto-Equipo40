import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay  # noqa: E402


def plot_regression(actual, predicted):
    """Compara los valores reales y estimados y muestra los residuales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(actual, predicted, gridsize=40, mincnt=1)
    upper = max(float(actual.max()), float(predicted.max()), 1)
    axes[0].plot([0, upper], [0, upper], "k--")
    axes[0].set(xlabel="Salidas reales", ylabel="Salidas estimadas", title="P1 · Validación")
    axes[1].hist(actual - predicted, bins=40)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Real − estimado", ylabel="Frecuencia", title="Residuales")
    fig.tight_layout()
    return fig


def plot_classification(actual, predicted, probabilities):
    """Matriz de confusión y curva precisión-recall de la misma validación."""
    confusion, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        actual, predicted, labels=[0, 1], display_labels=["Sin riesgo", "Riesgo"],
        ax=ax, colorbar=False,
    )
    ax.set_title("P2 · Matriz de confusión · Validación")
    confusion.tight_layout()

    precision_recall, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(actual, probabilities, ax=ax)
    ax.set_title("P2 · Precisión-recall · Validación")
    precision_recall.tight_layout()
    return {"confusion_matrix.png": confusion, "precision_recall.png": precision_recall}


def plot_cluster_profiles(dataset, features):
    """Compara los perfiles en escala común sin modificar las variables del modelo."""
    values = dataset[features]
    standardized = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    profile = standardized.groupby(dataset["cluster"]).mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    heatmap = ax.imshow(profile.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=35, ha="right")
    ax.set_yticks(range(len(profile)), profile.index)
    ax.set(ylabel="Grupo", title="P3 · Perfil medio estandarizado · Train")
    fig.colorbar(heatmap, ax=ax, label="Desviaciones estándar")
    fig.tight_layout()
    return fig


def plot_station_clusters(dataset, output_path):
    """Guarda la ubicación de las estaciones coloreada por cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        dataset["longitude"],
        dataset["latitude"],
        c=dataset["cluster"],
        s=40,
        alpha=0.75,
    )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("P3 - Clusters de estaciones MiBici")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
