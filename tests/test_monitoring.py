import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import (
    REPORT_COLUMNS,
    extract_column_drift,
    paired_daily_rmse_bootstrap,
    recommendation_from_ci,
    summarize_data_quality,
)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.5, -0.1, "candidate"),
        (0.1, 0.5, "production"),
        (-0.1, 0.1, "production (resultado inconcluso)"),
        (0.0, 0.0, "production (resultado inconcluso)"),
    ],
)
def test_recommendation_from_ci(low, high, expected):
    assert recommendation_from_ci(low, high) == expected


def test_bootstrap_is_reproducible_and_favors_better_candidate():
    dates = pd.Series(pd.date_range("2026-06-12", periods=14).repeat(2))
    actual = np.tile([10.0, 20.0], 14)
    production = actual + 2.0
    candidate = actual + 1.0

    first = paired_daily_rmse_bootstrap(
        dates, actual, production, candidate, seed=42
    )
    second = paired_daily_rmse_bootstrap(
        dates, actual, production, candidate, seed=42
    )

    assert first == second
    assert first["delta_rmse_candidate_minus_production"] < 0
    assert first["daily_cluster_bootstrap_95ci_high"] < 0


def test_recommendation_rejects_invalid_interval():
    with pytest.raises(ValueError, match="intervalo"):
        recommendation_from_ci(0.2, 0.1)


def make_drift_report(scores=None, omit_column=None):
    scores = scores or {}
    columns = [column for column in REPORT_COLUMNS if column != omit_column]
    return {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": column,
                    "threshold": 0.1,
                    "method": "test",
                },
                "value": scores.get(column, 0.02),
            }
            for column in columns
        ]
    }


def test_extract_column_drift_applies_threshold():
    report = make_drift_report({"departures_lag_1": 0.12})

    result = extract_column_drift(report)

    assert result["departures_lag_1"]["drift_detected"] is True
    assert result["station_id"]["drift_detected"] is False


def test_extract_column_drift_rejects_missing_metric():
    report = make_drift_report(omit_column=REPORT_COLUMNS[-1])

    with pytest.raises(ValueError, match="no devolvió métricas"):
        extract_column_drift(report)


def test_summarize_data_quality_counts_nulls_and_duplicates():
    frame = pd.DataFrame({"value": [1.0, 1.0, np.nan], "group": ["a", "a", "b"]})

    result = summarize_data_quality(frame)

    assert result["rows"] == 3
    assert result["duplicate_rows"] == 1
    assert result["nulls_by_column"]["value"] == 1