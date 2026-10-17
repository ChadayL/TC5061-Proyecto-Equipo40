import pandas as pd
from mibici.config import MIN_AGE, MAX_AGE


def remove_duplicate_trips(trips):
    return trips.drop_duplicates().reset_index(drop=True)


def replace_invalid_birth_years(trips, min_age, max_age):
    trips = trips.copy()
    birth_year = trips["Año_de_nacimiento"]
    approx_age = trips["Inicio_del_viaje"].dt.year - birth_year
    invalid_year = birth_year.notna() & (
        birth_year.le(0) | approx_age.lt(min_age) | approx_age.gt(max_age)
    )
    trips.loc[invalid_year, "Año_de_nacimiento"] = pd.NA
    return trips


def run(data):
    trips = remove_duplicate_trips(data["viajes"])
    trips = replace_invalid_birth_years(
        trips,
        min_age=MIN_AGE,
        max_age=MAX_AGE,
    )

    return {**data, "viajes": trips}
