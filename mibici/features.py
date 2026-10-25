# ============================================================
# P1 - REGRESIÓN
# ============================================================

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

def build_p3_features(daily_departures, arrival_totals):
    """Patrones de uso por estación a partir de salidas diarias y llegadas totales.

    Promedio y desviación de salidas se calculan sobre los días con salidas.
    Las estaciones sin salidas o sin llegadas quedan con cero.
    """
    grouped = daily_departures.groupby("estacion_id")

    departure_totals = grouped["salidas"].sum().rename("salidas_totales").reset_index()

    activity = (
        grouped["salidas"]
        .agg(salidas_promedio="mean", salidas_std="std")
        .reset_index()
    )

    weekend_share = (
        daily_departures["salidas"]
        .where(daily_departures["fin_semana"].eq(1), 0)
        .groupby(daily_departures["estacion_id"])
        .sum()
        .div(grouped["salidas"].sum())
        .rename("proporcion_fin_semana")
        .reset_index()
    )

    features = (
        departure_totals.merge(arrival_totals, on="estacion_id", how="outer")
        .merge(activity, on="estacion_id", how="left")
        .merge(weekend_share, on="estacion_id", how="left")
        .fillna(0)
    )

    features["viajes_totales"] = features["salidas_totales"] + features["llegadas_totales"]

    # Positivo: llegaron más bicicletas de las que salieron.
    features["balance"] = features["llegadas_totales"] - features["salidas_totales"]
    features["ratio_llegadas_salidas"] = features["llegadas_totales"] / (
        features["salidas_totales"] + 1
    )

    return features
