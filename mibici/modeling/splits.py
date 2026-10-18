import math

import pandas as pd


def define_time_periods(
    dates,
    train_ratio,
    validation_ratio,
    test_ratio,
):
    ratios = [train_ratio, validation_ratio, test_ratio]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Las proporciones deben ser positivas.")

    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Las proporciones deben sumar 1.")

    available_dates = pd.DatetimeIndex(
        pd.to_datetime(dates).dropna()
    ).normalize()

    if available_dates.empty:
        raise ValueError("No hay fechas disponibles.")

    calendar = pd.date_range(
        available_dates.min(),
        available_dates.max(),
        freq="D",
    )

    train_days = int(len(calendar) * train_ratio)
    validation_days = int(len(calendar) * validation_ratio)
    test_days = len(calendar) - train_days - validation_days

    if min(train_days, validation_days, test_days) < 1:
        raise ValueError("Cada conjunto debe tener al menos un día.")

    return {
        "start": calendar[0],
        "validation_start": calendar[train_days],
        "test_start": calendar[train_days + validation_days],
        "end": calendar[-1],
    }