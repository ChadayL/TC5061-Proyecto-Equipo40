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
