import pandas as pd


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def convert_birth_year(trips):
    trips = trips.copy()
    trips["Año_de_nacimiento"] = trips["Año_de_nacimiento"].astype("Int64")
    return trips


def convert_gender(trips):
    trips = trips.copy()
    trips["Genero"] = trips["Genero"].astype("string")
    return trips


def convert_dates(trips):
    trips = trips.copy()

    for column in ["Inicio_del_viaje", "Fin_del_viaje"]:
        trips[column] = pd.to_datetime(
            trips[column],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )

    return trips


def convert_station_name(stations):
    stations = stations.copy()
    stations["name"] = stations["name"].astype("string")
    return stations


def run(data):
    trips = normalize_columns(data["viajes"])
    trips = convert_birth_year(trips)
    trips = convert_gender(trips)
    trips = convert_dates(trips)

    stations = normalize_columns(data["estaciones"])
    stations = convert_station_name(stations)

    return {"viajes": trips, "estaciones": stations}