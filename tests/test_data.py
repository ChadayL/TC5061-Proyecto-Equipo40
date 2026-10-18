import numpy as np
import pandas as pd
import pytest

from mibici.dataset import load_raw
from mibici.features import (
    build_p1_features,
    build_p2_features,
    build_p3_features,
)
from mibici.modeling import p2, p3
from mibici.pipeline import load_stage

# Muestra de los archivos reales para probar la carga sin leer toda la partición.
SAMPLE_ROWS = 1_000


def make_trips(rows):
    """Viajes mínimos: (inicio, fin, origen, destino)."""
    trips = pd.DataFrame(
        rows,
        columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
    )
    trips["Inicio_del_viaje"] = pd.to_datetime(trips["Inicio_del_viaje"])
    trips["Fin_del_viaje"] = pd.to_datetime(trips["Fin_del_viaje"])
    return trips


def make_stations(capacities):
    """Catálogo mínimo: {id: capacidad}."""
    ids = list(capacities)
    return pd.DataFrame({
        "id": ids,
        "name": [f"Estación {i}" for i in ids],
        "latitude": [20.6 + i / 100 for i in ids],
        "longitude": [-103.3 - i / 100 for i in ids],
        "dpcapacity": list(capacities.values()),
    })


# ============================================================
# CARGA Y PIPELINE (archivos reales, muestra pequeña)
# ============================================================

def test_load_raw():
    datos = load_raw(nrows=SAMPLE_ROWS)

    assert set(datos) == {"viajes", "estaciones"}
    assert len(datos["viajes"]) == SAMPLE_ROWS
    assert len(datos["estaciones"]) > 0


@pytest.mark.parametrize("stage", ["raw", "structure", "quality_fixes", "common"])
def test_pipeline_stages(stage):
    datos = load_stage(stage, nrows=SAMPLE_ROWS)

    assert set(datos) == {"viajes", "estaciones"}
    assert 0 < len(datos["viajes"]) <= SAMPLE_ROWS


# ============================================================
# P1
# ============================================================

def make_daily_demand():
    """Dos estaciones, nueve días continuos (2025-01-01 es miércoles)."""
    dates = pd.date_range("2025-01-01", periods=9, freq="D")
    return pd.DataFrame({
        "station_id": [1] * 9 + [2] * 9,
        "date": list(dates) * 2,
        "departures": list(range(9)) + list(range(10, 19)),
    })


def test_p1_features_columns_and_rows():
    daily = make_daily_demand()
    features = build_p1_features(daily)

    assert len(features) == len(daily)
    for column in [
        "day_of_week",
        "departures_lag_1",
        "departures_lag_7",
        "departures_mean_7",
    ]:
        assert column in features.columns
    assert features.loc[features["date"].eq("2025-01-01"), "day_of_week"].eq(2).all()


def test_p1_lags_within_station():
    features = build_p1_features(make_daily_demand()).set_index(["station_id", "date"])

    # Estación 2, 2025-01-08: salidas 17
    row = features.loc[(2, pd.Timestamp("2025-01-08"))]
    assert row["departures_lag_1"] == 16
    assert row["departures_lag_7"] == 10
    # Promedio de los siete días anteriores, sin incluir el día evaluado
    assert row["departures_mean_7"] == pytest.approx(np.mean(range(10, 17)))

    # El primer día de la estación 2 no toma valores de la estación 1
    assert pd.isna(features.loc[(2, pd.Timestamp("2025-01-01")), "departures_lag_1"])


def test_p1_missing_history_is_not_zero():
    features = build_p1_features(make_daily_demand())

    assert features["departures_lag_1"].isna().sum() == 2
    assert features["departures_lag_7"].isna().sum() == 14
    assert features["departures_mean_7"].isna().sum() == 14


# ============================================================
# P2
# ============================================================

P2_STATIONS = make_stations({1: 10, 2: 4, 3: 0})

P2_TRIPS = make_trips([
    ("2025-01-01 08:00", "2025-01-01 08:20", 1, 2),
    ("2025-01-01 08:05", "2025-01-01 08:25", 1, 2),
    ("2025-01-01 08:10", "2025-01-01 08:30", 1, 2),
    # Cruza la medianoche: sale el 1 y llega el 2
    ("2025-01-01 23:50", "2025-01-02 00:10", 2, 1),
    ("2025-01-03 09:00", "2025-01-03 09:15", 2, 2),
])


def test_p2_daily_flows():
    daily = p2.build_daily_flows(P2_TRIPS, P2_STATIONS).set_index(["fecha", "estacion_id"])

    # Todas las estaciones del catálogo por todos los días, con ceros
    assert len(daily) == 3 * 3
    assert daily.loc[(pd.Timestamp("2025-01-01"), 1), "salidas"] == 3
    assert daily.loc[(pd.Timestamp("2025-01-01"), 2), "salidas"] == 1
    assert daily.loc[(pd.Timestamp("2025-01-01"), 2), "llegadas"] == 3
    # La llegada se cuenta por Fin_del_viaje
    assert daily.loc[(pd.Timestamp("2025-01-02"), 1), "llegadas"] == 1
    assert daily.loc[(pd.Timestamp("2025-01-02"), 3), ["salidas", "llegadas"]].eq(0).all()


def test_p2_station_capacity_excludes_zero():
    daily = p2.build_daily_flows(P2_TRIPS, P2_STATIONS)
    with_capacity = p2.add_station_capacity(daily, P2_STATIONS)

    assert set(with_capacity["estacion_id"]) == {1, 2}
    assert with_capacity["capacity"].gt(0).all()


def test_p2_features_and_target():
    daily = p2.add_station_capacity(p2.build_daily_flows(P2_TRIPS, P2_STATIONS), P2_STATIONS)
    dataset = p2.build_risk_target(build_p2_features(daily), threshold=0.5)
    dataset = dataset.set_index(["estacion_id", "fecha"])

    station_2 = dataset.loc[(2, pd.Timestamp("2025-01-01"))]
    assert station_2["balance"] == -2
    assert station_2["presion_capacidad"] == pytest.approx(0.5)
    assert station_2["riesgo_actual"] == 1

    station_1 = dataset.loc[(1, pd.Timestamp("2025-01-01"))]
    assert station_1["balance"] == 3
    assert station_1["presion_capacidad"] == pytest.approx(0.3)
    assert station_1["riesgo_actual"] == 0
    # Objetivo: riesgo del día siguiente de la misma estación (balance -1 / 10)
    assert station_1["riesgo_manana"] == 0

    # El último día no tiene objetivo
    assert pd.isna(dataset.loc[(1, pd.Timestamp("2025-01-03")), "riesgo_manana"])


def test_p2_lags_within_station():
    dates = pd.date_range("2025-01-01", periods=9, freq="D")
    daily = pd.DataFrame({
        "fecha": list(dates) * 2,
        "estacion_id": [1] * 9 + [2] * 9,
        "salidas": list(range(9)) * 2,
        "llegadas": 0,
        "capacity": [10] * 9 + [5] * 9,
    })
    features = build_p2_features(daily).set_index(["estacion_id", "fecha"])

    row = features.loc[(1, pd.Timestamp("2025-01-09"))]
    assert row["balance_lag_1"] == 7
    assert row["balance_lag_7"] == 1
    assert row["presion_lag_1"] == pytest.approx(0.7)
    # Promedio de presión de los siete días anteriores (1..7) / 10
    assert row["presion_promedio_7d"] == pytest.approx(0.4)

    assert pd.isna(features.loc[(2, pd.Timestamp("2025-01-01")), "balance_lag_1"])
    assert features["balance_lag_7"].isna().sum() == 14


def test_p2_period_dataset_by_target_date():
    # Un viaje diario de la estación 1 a la 2 durante diez días
    trips = make_trips([
        (f"2025-01-{day:02d} 08:00", f"2025-01-{day:02d} 08:30", 1, 2) for day in range(1, 11)
    ])
    start, end_exclusive = pd.Timestamp("2025-01-09"), pd.Timestamp("2025-01-11")

    dataset = p2.build_period_dataset(trips, P2_STATIONS, start, end_exclusive)

    assert dataset["fecha_objetivo"].ge(start).all()
    assert dataset["fecha_objetivo"].lt(end_exclusive).all()
    assert (dataset["fecha_objetivo"] - dataset["fecha"]).eq(pd.Timedelta(days=1)).all()
    # Estaciones 1 y 2 (la 3 tiene capacidad 0), predicciones del 8 y 9 de enero
    assert len(dataset) == 4
    assert dataset[p2.HISTORY_FEATURES].notna().all().all()
    assert set(dataset[p2.TARGET].unique()) <= {0, 1}


# ============================================================
# P3
# ============================================================

P3_STATIONS = make_stations({1: 10, 2: 4, 3: 0, 4: 8})

P3_TRIPS = make_trips([
    ("2025-01-01 08:00", "2025-01-01 08:10", 1, 2),
    ("2025-01-01 09:00", "2025-01-01 09:10", 1, 2),
    # 2025-01-04 es sábado
    ("2025-01-04 09:00", "2025-01-04 09:30", 1, 2),
    ("2025-01-02 10:00", "2025-01-02 10:20", 2, 1),
    ("2025-01-02 12:00", "2025-01-02 12:10", 2, 4),
    ("2025-01-03 12:00", "2025-01-03 12:10", 2, 3),
])


def build_p3_dataset():
    features = build_p3_features(
        p3.build_daily_departures(P3_TRIPS),
        p3.build_arrival_totals(P3_TRIPS),
    )
    return p3.add_station_info(features, P3_STATIONS)


def test_p3_one_row_per_eligible_station():
    dataset = build_p3_dataset()

    assert dataset["estacion_id"].is_unique
    # La 3 tiene capacidad 0; la 4 solo recibe viajes y se conserva
    assert set(dataset["estacion_id"]) == {1, 2, 4}
    assert dataset["capacity"].gt(0).all()


def test_p3_usage_patterns():
    dataset = build_p3_dataset().set_index("estacion_id")

    station_1 = dataset.loc[1]
    assert station_1["salidas_totales"] == 3
    assert station_1["llegadas_totales"] == 1
    # Promedio y desviación sobre los días con salidas (2 y 1)
    assert station_1["salidas_promedio"] == pytest.approx(1.5)
    assert station_1["salidas_std"] == pytest.approx(np.std([2, 1], ddof=1))
    assert station_1["proporcion_fin_semana"] == pytest.approx(1 / 3)
    assert station_1["viajes_totales"] == 4
    assert station_1["balance"] == -2
    assert station_1["ratio_llegadas_salidas"] == pytest.approx(1 / 4)

    # Estación sin salidas: queda en cero en lugar de faltante
    station_4 = dataset.loc[4]
    assert station_4[["salidas_totales", "salidas_promedio", "salidas_std"]].eq(0).all()
    assert station_4["llegadas_totales"] == 1
    assert station_4["balance"] == 1
