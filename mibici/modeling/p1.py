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