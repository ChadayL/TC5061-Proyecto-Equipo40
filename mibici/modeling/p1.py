from mibici.pipeline import load_stage
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from mibici.data_preparation.encoding import build_encoder
from mibici.data_preparation.scaling import build_scaler



CATEGORICAL_FEATURES = ["station_id", "day_of_week"]

NUMERIC_FEATURES = [
    "departures_lag_1",
    "departures_lag_7",
    "departures_mean_7",
]

TARGET = "departures"

def build_training_daily_data(trips, start, end_exclusive):
    """Cuenta salidas desde la primera actividad hasta el corte."""
    activity = pd.concat(
        [
            trips[["Origen_Id", "Inicio_del_viaje"]].rename(
                columns={
                    "Origen_Id": "station_id",
                    "Inicio_del_viaje": "date",
                }
            ),
            trips[["Destino_Id", "Fin_del_viaje"]].rename(
                columns={
                    "Destino_Id": "station_id",
                    "Fin_del_viaje": "date",
                }
            ),
        ],
        ignore_index=True,
    )

    activity["date"] = activity["date"].dt.normalize()

    activity = activity.loc[
        activity["date"].ge(start)
        & activity["date"].lt(end_exclusive)
    ]

    first_activity = activity.groupby("station_id")["date"].min()

    calendar = (
        first_activity.apply(
            lambda first_date: pd.date_range(
                first_date,
                end_exclusive - pd.Timedelta(days=1),
                freq="D",
            )
        )
        .rename("date")
        .explode()
        .reset_index()
    )
    calendar["date"] = pd.to_datetime(calendar["date"])

    departure_counts = (
        trips.assign(date=trips["Inicio_del_viaje"].dt.normalize())
        .groupby(["Origen_Id", "date"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(columns={"Origen_Id": "station_id"})
    )

    result = calendar.merge(
        departure_counts,
        on=["station_id", "date"],
        how="left",
        validate="one_to_one",
    )

    result["departures"] = (
        result["departures"].fillna(0).astype("int64")
    )

    return result.sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

def build_daily_departures(trips):
    """Construye el calendario exploratorio diario de P1."""
    departures = (
        trips.assign(date=trips["Inicio_del_viaje"].dt.normalize())
        .groupby(["Origen_Id", "date"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(columns={"Origen_Id": "station_id"})
    )

    # Actividad en ambas direcciones.
    departures_activity = pd.DataFrame({
        "station_id": trips["Origen_Id"],
        "date": trips["Inicio_del_viaje"].dt.normalize(),
    })

    arrivals_activity = pd.DataFrame({
        "station_id": trips["Destino_Id"],
        "date": trips["Fin_del_viaje"].dt.normalize(),
    })

    activity = pd.concat(
        [departures_activity, arrivals_activity],
        ignore_index=True,
    )

    # Evitar extender el calendario por devoluciones fuera del periodo.
    period_start = departures["date"].min()
    period_end = departures["date"].max()

    activity = activity.loc[
        activity["date"].between(period_start, period_end)
    ]

    spans = activity.groupby("station_id")["date"].agg(
        first_activity="min",
        last_activity="max",
    )

    calendar = spans.apply(
        lambda row: pd.date_range(
            row["first_activity"],
            row["last_activity"],
            freq="D",
        ),
        axis=1,
    )

    calendar = (
        calendar.rename("date")
        .explode()
        .reset_index()
    )
    calendar["date"] = pd.to_datetime(calendar["date"])

    daily_demand = calendar.merge(
        departures,
        on=["station_id", "date"],
        how="left",
        validate="one_to_one",
    )

    daily_demand["departures"] = (
        daily_demand["departures"].fillna(0).astype("int64")
    )

    return daily_demand.sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

def predict_departures(model, features):
    """Predice salidas y limita los resultados negativos a cero."""
    predictions = model.predict(features)
    return predictions.clip(min=0)

def load_p1_data():
    data = load_stage("common")
    return build_daily_departures(data["viajes"])

def build_model_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", build_encoder(), CATEGORICAL_FEATURES),
            ("numeric", build_scaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )