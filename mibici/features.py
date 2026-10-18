import pandas as pd


def add_calendar_features(daily_demand):
    result = daily_demand.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    return result


def add_lag_features(daily_demand):
    result = daily_demand.sort_values(["station_id", "date"]).copy()
    for lag in (1, 7):
        result[f"departures_lag_{lag}"] = result.groupby("station_id")[
            "departures"
        ].shift(lag)
    return result


def add_rolling_features(daily_demand):
    result = daily_demand.sort_values(["station_id", "date"]).copy()
    result["departures_mean_7"] = result.groupby("station_id")["departures"].transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean()
    )
    return result


def build_p1_features(daily_demand):
    result = add_calendar_features(daily_demand)
    result = add_lag_features(result)
    result = add_rolling_features(result)
    return result


# ============================================================
# P2 - CLASIFICACIÓN
# ============================================================

def build_p2_features(daily):
    """Variables de P2 a partir de salidas, llegadas y capacidad por estación y día.

    Los rezagos y promedios usan solo días anteriores dentro de cada estación.
    """
    result = daily.sort_values(["estacion_id", "fecha"]).copy()

    # Calendario (lunes = 0, domingo = 6)
    result["dia_semana"] = result["fecha"].dt.dayofweek
    result["fin_semana"] = result["dia_semana"].ge(5).astype(int)

    # Positivo: salieron más bicicletas de las que llegaron.
    result["balance"] = result["salidas"] - result["llegadas"]
    result["presion_capacidad"] = result["balance"].abs() / result["capacity"]

    grouped = result.groupby("estacion_id")
    result["balance_lag_1"] = grouped["balance"].shift(1)
    result["balance_lag_7"] = grouped["balance"].shift(7)
    result["presion_lag_1"] = grouped["presion_capacidad"].shift(1)
    result["presion_promedio_7d"] = grouped["presion_capacidad"].transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).mean()
    )

    return result


# ============================================================
# P3 - CLUSTERING DE ESTACIONES
# ============================================================

def build_p3_features(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para P3.

    Pregunta:
    ¿Qué grupos de estaciones comparten patrones de uso
    y dónde se ubican en la ciudad?

    Definición:
    - Tipo de problema: Clustering
    - Granularidad: estación
    - No existe variable objetivo
    - Los patrones se construyen a partir del uso histórico
      de cada estación.

    Las coordenadas geográficas se conservan para visualizar
    posteriormente la ubicación de los clusters.
    """

    # ========================================================
    # 1. COPIAR DATOS
    # ========================================================

    df = viajes.copy()
    estaciones = estaciones.copy()

    # ========================================================
    # 2. PREPARAR FECHA
    # ========================================================

    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"],
        errors="coerce",
    )

    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Inicio_del_viaje",
            "Origen_Id",
            "Destino_Id",
        ]
    )

    df["fecha"] = (
        df["Inicio_del_viaje"]
        .dt.normalize()
    )

    df["fin_semana"] = (
        df["Inicio_del_viaje"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    # ========================================================
    # 3. SALIDAS POR ESTACIÓN
    # ========================================================

    salidas = (
        df.groupby("Origen_Id")
        .size()
        .rename("salidas_totales")
        .reset_index()
        .rename(
            columns={
                "Origen_Id": "estacion_id"
            }
        )
    )

    # ========================================================
    # 4. LLEGADAS POR ESTACIÓN
    # ========================================================

    llegadas = (
        df.groupby("Destino_Id")
        .size()
        .rename("llegadas_totales")
        .reset_index()
        .rename(
            columns={
                "Destino_Id": "estacion_id"
            }
        )
    )

    # ========================================================
    # 5. ACTIVIDAD DIARIA
    # ========================================================

    salidas_dia = (
        df.groupby(
            ["fecha", "Origen_Id"]
        )
        .size()
        .rename("salidas")
        .reset_index()
    )

    actividad = (
        salidas_dia
        .groupby("Origen_Id")["salidas"]
        .agg(
            salidas_promedio="mean",
            salidas_std="std",
        )
        .reset_index()
        .rename(
            columns={
                "Origen_Id": "estacion_id"
            }
        )
    )

    # ========================================================
    # 6. USO EN FIN DE SEMANA
    # ========================================================

    uso_fin_semana = (
        df.groupby("Origen_Id")
        ["fin_semana"]
        .mean()
        .rename("proporcion_fin_semana")
        .reset_index()
        .rename(
            columns={
                "Origen_Id": "estacion_id"
            }
        )
    )

    # ========================================================
    # 7. UNIR PATRONES DE USO
    # ========================================================

    features = (
        salidas
        .merge(
            llegadas,
            on="estacion_id",
            how="outer",
        )
        .merge(
            actividad,
            on="estacion_id",
            how="left",
        )
        .merge(
            uso_fin_semana,
            on="estacion_id",
            how="left",
        )
    )

    features = features.fillna(0)

    # ========================================================
    # 8. VARIABLES DE BALANCE
    # ========================================================

    features["viajes_totales"] = (
        features["salidas_totales"]
        + features["llegadas_totales"]
    )

    features["balance"] = (
        features["llegadas_totales"]
        - features["salidas_totales"]
    )

    features["ratio_llegadas_salidas"] = (
        features["llegadas_totales"]
        / (
            features["salidas_totales"]
            + 1
        )
    )

    # ========================================================
    # 9. INFORMACIÓN DE ESTACIONES
    # ========================================================

    estaciones_p3 = estaciones[
        [
            "id",
            "latitude",
            "longitude",
            "dpcapacity",
        ]
    ].copy()

    estaciones_p3 = estaciones_p3.rename(
        columns={
            "id": "estacion_id",
            "dpcapacity": "capacity",
        }
    )

    features = features.merge(
        estaciones_p3,
        on="estacion_id",
        how="left",
    )

    # ========================================================
    # 10. LIMPIEZA FINAL
    # ========================================================

    columnas_numericas = [
        "salidas_totales",
        "llegadas_totales",
        "salidas_promedio",
        "salidas_std",
        "proporcion_fin_semana",
        "viajes_totales",
        "balance",
        "ratio_llegadas_salidas",
        "latitude",
        "longitude",
        "capacity",
    ]

    for columna in columnas_numericas:
        features[columna] = pd.to_numeric(
            features[columna],
            errors="coerce",
        )

    features = features.dropna(
        subset=[
            "latitude",
            "longitude",
            "capacity",
        ]
    )

    features = features[
        features["capacity"] > 0
    ]

    features = features.reset_index(
        drop=True
    )

    return features
