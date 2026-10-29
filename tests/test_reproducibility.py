import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from mibici.modeling.p2 import build_p2_model


def test_random_forest_reproducible_predictions_and_metric():
    features = pd.DataFrame(
        {
            "salidas": [0, 1, 4, 5, 2, 8, 7, 1, 6, 3, 9, 2],
            "llegadas": [3, 2, 1, 0, 5, 1, 2, 4, 1, 3, 0, 6],
            "capacity": [10] * 12,
        }
    )
    target = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0])

    first = build_p2_model(n_estimators=20, max_depth=4, random_state=42)
    second = build_p2_model(n_estimators=20, max_depth=4, random_state=42)
    first.fit(features, target)
    second.fit(features, target)

    first_predictions = first.predict(features)
    second_predictions = second.predict(features)
    assert np.array_equal(first_predictions, second_predictions)
    assert f1_score(target, first_predictions) == f1_score(target, second_predictions)
