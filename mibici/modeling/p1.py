import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from mibici.config import P1_RIDGE_ALPHA
from mibici.data_preparation.encoding import build_encoder
from mibici.data_preparation.scaling import build_scaler
from mibici.dataset import get_split_periods
from mibici.features import build_p1_features
from mibici.pipeline import load_stage

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
        activity["date"].ge(start) & activity["date"].lt(end_exclusive)
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

    result["departures"] = result["departures"].fillna(0).astype("int64")

    return result.sort_values(["station_id", "date"]).reset_index(drop=True)


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
    departures_activity = pd.DataFrame(
        {
            "station_id": trips["Origen_Id"],
            "date": trips["Inicio_del_viaje"].dt.normalize(),
        }
    )

    arrivals_activity = pd.DataFrame(
        {
            "station_id": trips["Destino_Id"],
            "date": trips["Fin_del_viaje"].dt.normalize(),
        }
    )

    activity = pd.concat(
        [departures_activity, arrivals_activity],
        ignore_index=True,
    )

    # Evitar extender el calendario por devoluciones fuera del periodo.
    period_start = departures["date"].min()
    period_end = departures["date"].max()

    activity = activity.loc[activity["date"].between(period_start, period_end)]

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

    calendar = calendar.rename("date").explode().reset_index()
    calendar["date"] = pd.to_datetime(calendar["date"])

    daily_demand = calendar.merge(
        departures,
        on=["station_id", "date"],
        how="left",
        validate="one_to_one",
    )

    daily_demand["departures"] = daily_demand["departures"].fillna(0).astype("int64")

    return daily_demand.sort_values(["station_id", "date"]).reset_index(drop=True)


def predict_departures(model, features):
    """Predice salidas y limita los resultados negativos a cero."""
    predictions = model.predict(features)
    return predictions.clip(min=0)


def build_evaluation_daily_data(
    trips,
    station_ids,
    start,
    end_exclusive,
):
    calendar = pd.MultiIndex.from_product(
        [
            station_ids,
            pd.date_range(
                start,
                end_exclusive - pd.Timedelta(days=1),
                freq="D",
            ),
        ],
        names=["station_id", "date"],
    ).to_frame(index=False)

    counts = (
        trips.assign(date=trips["Inicio_del_viaje"].dt.normalize())
        .groupby(["Origen_Id", "date"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(columns={"Origen_Id": "station_id"})
    )

    result = calendar.merge(
        counts,
        on=["station_id", "date"],
        how="left",
        validate="one_to_one",
    )

    result["departures"] = result["departures"].fillna(0).astype("int64")

    return result.sort_values(["station_id", "date"]).reset_index(drop=True)


def prepare_p1_data():
    """Prepara train y validation para predicciones diarias a un paso."""
    periods = get_split_periods()
    train_start, train_end = periods["train"]
    validation_start, validation_end = periods["validation"]

    # Entrenamiento.
    train_raw = load_stage("common", split="train")
    train_trips = train_raw["viajes"]

    train_daily = build_training_daily_data(
        train_trips,
        start=train_start,
        end_exclusive=train_end,
    )

    train_features = build_p1_features(train_daily)

    train_ready = train_features.dropna(subset=NUMERIC_FEATURES).copy()

    if train_ready.empty:
        raise ValueError("No hay entrenamiento con historia suficiente.")

    known_stations = sorted(train_ready["station_id"].unique())

    # Validación: mismas estaciones y calendario completo.
    validation_raw = load_stage("common", split="validation")
    validation_trips = validation_raw["viajes"]

    validation_daily = build_evaluation_daily_data(
        validation_trips,
        station_ids=known_stations,
        start=validation_start,
        end_exclusive=validation_end,
    )

    # Contexto anterior al inicio de validación.
    history = train_daily.loc[
        train_daily["station_id"].isin(known_stations)
        & train_daily["date"].ge(validation_start - pd.Timedelta(days=7))
        & train_daily["date"].lt(validation_start)
    ].copy()

    context = pd.concat(
        [history, validation_daily],
        ignore_index=True,
    )

    context_features = build_p1_features(context)

    validation_ready = context_features.loc[
        context_features["date"].ge(validation_start)
        & context_features["date"].lt(validation_end)
    ].copy()

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    if train_ready[feature_columns + [TARGET]].isna().any().any():
        raise ValueError("Hay faltantes en entrenamiento.")

    if validation_ready[feature_columns + [TARGET]].isna().any().any():
        raise ValueError("Hay faltantes en validación.")

    included = validation_trips["Origen_Id"].isin(known_stations)

    return {
        "train": train_ready,
        "validation": validation_ready,
        "periods": periods,
        "summary": {
            "train_rows_excluded": len(train_features) - len(train_ready),
            "known_stations": len(known_stations),
            "validation_trips_excluded": int((~included).sum()),
        },
    }


def load_p1_data():
    data = load_stage("common")
    return build_daily_departures(data["viajes"])


def build_model_pipeline(fit_intercept=True, model_name="linear_regression", alpha=P1_RIDGE_ALPHA):
    if model_name == "linear_regression":
        regressor = LinearRegression(fit_intercept=fit_intercept)
    elif model_name == "ridge":
        regressor = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="lsqr")
    else:
        raise ValueError(f"Modelo no disponible para P1: {model_name!r}")

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
            ("regressor", regressor),
        ]
    )
