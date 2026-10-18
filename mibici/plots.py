import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


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
