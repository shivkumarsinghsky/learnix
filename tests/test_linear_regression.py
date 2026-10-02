import json
from pathlib import Path

import pytest

from learnix import (
    LinearRegression,
    TrainingDivergedError,
    fit_closed_form,
    load_csv_xy,
    r2_score,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "salary.csv"
XS = [1.0, 2.0, 3.0, 4.0]
YS = [35000.0, 40000.0, 45000.0, 50000.0]  # salary = 5000 * x + 30000


@pytest.fixture(scope="module")
def salary_data() -> tuple[list[float], list[float]]:
    return load_csv_xy(DATA, "experience_years", "salary")


def test_analytic_gradients_match_numerical_derivatives() -> None:
    m, b, h = 1234.0, 5678.0, 1e-3
    grad_m, grad_b = LinearRegression.gradients(XS, YS, m, b)
    loss = LinearRegression.loss
    numeric_m = (loss(XS, YS, m + h, b) - loss(XS, YS, m - h, b)) / (2 * h)
    numeric_b = (loss(XS, YS, m, b + h) - loss(XS, YS, m, b - h)) / (2 * h)
    assert grad_m == pytest.approx(numeric_m, rel=1e-6)
    assert grad_b == pytest.approx(numeric_b, rel=1e-6)


def test_gradient_descent_recovers_the_exact_line() -> None:
    model = LinearRegression(learning_rate=0.05, epochs=5000).fit(XS, YS)
    assert model.m == pytest.approx(5000, rel=1e-4)
    assert model.b == pytest.approx(30000, rel=1e-4)
    assert model.predict([5]) == [pytest.approx(55000, rel=1e-4)]


def test_loss_never_increases_with_a_stable_learning_rate() -> None:
    history = LinearRegression(learning_rate=0.05, epochs=500).fit(XS, YS).loss_history
    assert all(later <= earlier for earlier, later in zip(history, history[1:], strict=False))


def test_original_tiny_learning_rate_barely_trains() -> None:
    # Why the default learning rate changed: 1e-6 for 100 epochs stays far from the answer.
    model = LinearRegression(learning_rate=0.000001, epochs=100).fit(XS, YS)
    assert model.m < 100
    assert model.predict([5])[0] < 1000


def test_too_large_learning_rate_raises_instead_of_returning_nan() -> None:
    with pytest.raises(TrainingDivergedError, match="learning_rate"):
        LinearRegression(learning_rate=0.5, epochs=5000).fit(XS, YS)


def test_standardized_training_matches_closed_form(
    salary_data: tuple[list[float], list[float]],
) -> None:
    xs, ys = salary_data
    model = LinearRegression(learning_rate=0.1, epochs=2000, standardize=True).fit(xs, ys)
    exact_m, exact_b = fit_closed_form(xs, ys)
    assert model.m == pytest.approx(exact_m, rel=1e-6)
    assert model.b == pytest.approx(exact_b, rel=1e-6)
    assert r2_score(ys, model.predict(xs)) > 0.95


def test_early_stopping_uses_fewer_epochs(salary_data: tuple[list[float], list[float]]) -> None:
    xs, ys = salary_data
    model = LinearRegression(learning_rate=0.1, epochs=2000, standardize=True, tolerance=1e-6)
    model.fit(xs, ys)
    assert len(model.loss_history) < 2000


def test_matches_scikit_learn(salary_data: tuple[list[float], list[float]]) -> None:
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    xs, ys = salary_data
    reference = sklearn_linear.LinearRegression().fit([[x] for x in xs], ys)
    model = LinearRegression(learning_rate=0.1, epochs=2000, standardize=True).fit(xs, ys)
    assert model.m == pytest.approx(float(reference.coef_[0]), rel=1e-6)
    assert model.b == pytest.approx(float(reference.intercept_), rel=1e-6)


def test_save_and_load_round_trip_as_plain_json(tmp_path: Path) -> None:
    model = LinearRegression(learning_rate=0.05, epochs=3000).fit(XS, YS)
    path = model.save(tmp_path / "model.json")
    data = json.loads(path.read_text())
    assert data["format"] == "learnix.LinearRegression"
    reloaded = LinearRegression.load(path)
    assert (reloaded.m, reloaded.b) == (model.m, model.b)
    assert reloaded.learning_rate == 0.05


def test_load_rejects_files_that_are_not_learnix_models(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"m": 1, "b": 2}))
    with pytest.raises(ValueError, match="not a learnix"):
        LinearRegression.load(other)


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0}, {"epochs": 0}, {"tolerance": -1}])
def test_invalid_hyperparameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LinearRegression(**kwargs)  # type: ignore[arg-type]


def test_standardize_rejects_constant_x() -> None:
    with pytest.raises(ValueError, match="identical"):
        LinearRegression(standardize=True).fit([3.0, 3.0], [1.0, 2.0])
