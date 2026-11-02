import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_model(seed: int = 42):
    """Entrena un modelo determinista usando una semilla fija."""
    rng = np.random.default_rng(42)

    X = rng.normal(size=(500, 5))
    y = (
        3 * X[:, 0]
        - 2 * X[:, 1]
        + 0.5 * X[:, 2]
        + rng.normal(scale=0.1, size=500)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=6,
        random_state=seed,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, rmse


def test_training_is_reproducible():
    """Dos entrenamientos con la misma semilla deben dar el mismo resultado."""
    predictions_1, rmse_1 = train_model(seed=42)
    predictions_2, rmse_2 = train_model(seed=42)

    np.testing.assert_allclose(
        predictions_1,
        predictions_2,
        rtol=0,
        atol=1e-12,
    )

    assert np.isclose(
        rmse_1,
        rmse_2,
        rtol=0,
        atol=1e-12,
    )