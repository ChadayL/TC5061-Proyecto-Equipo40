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
        # Permite recibir una columna ya convertida.
        if pd.api.types.is_datetime64_any_dtype(trips[column]):
            continue

        original = trips[column].astype("string")
        slash_dates = original.str.contains("/", regex=False, na=False)

        parsed = pd.Series(
            pd.NaT,
            index=trips.index,
            dtype="datetime64[ns]",
        )

        parsed.loc[slash_dates] = pd.to_datetime(
            original.loc[slash_dates],
            format="%d/%m/%Y %H:%M",
            errors="raise",
        )

        parsed.loc[~slash_dates] = pd.to_datetime(
            original.loc[~slash_dates],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )

        trips[column] = parsed

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