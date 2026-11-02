import numpy as np

from mibici.modeling.p3 import build_p3_model, select_best_k


def test_select_best_k_returns_k_with_highest_score():
    assert select_best_k({2: 0.42, 3: 0.81, 4: 0.67}) == 3


def test_p3_clustering_is_reproducible():
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )

    first = build_p3_model(2).fit_predict(features)
    second = build_p3_model(2).fit_predict(features)

    np.testing.assert_array_equal(first, second)
    assert len(set(first)) == 2
