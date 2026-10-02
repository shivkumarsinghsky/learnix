import math

import pytest

from learnix import fit_closed_form, mean_squared_error, r2_score, residuals


def test_residuals_are_actual_minus_predicted() -> None:
    assert residuals([35000, 40000], [11000, 12000]) == [24000, 28000]


def test_mean_squared_error_matches_hand_calculation() -> None:
    # lesson 3 numbers: residuals 24000, 28000, 32000, 36000
    actual = [35000, 40000, 45000, 50000]
    predicted = [11000, 12000, 13000, 14000]
    expected = (24000**2 + 28000**2 + 32000**2 + 36000**2) / 4
    assert mean_squared_error(actual, predicted) == expected


def test_r2_is_one_for_perfect_fit_and_zero_for_mean_model() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    assert r2_score(actual, actual) == 1.0
    assert r2_score(actual, [2.5] * 4) == 0.0


def test_r2_undefined_for_constant_targets() -> None:
    with pytest.raises(ValueError, match="undefined"):
        r2_score([3.0, 3.0], [3.0, 3.0])


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [([], [], "at least one"), ([1.0], [1.0, 2.0], "same length"), ([1.0], [math.nan], "finite")],
)
def test_invalid_inputs_are_rejected(
    actual: list[float], predicted: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mean_squared_error(actual, predicted)


def test_closed_form_recovers_exact_line() -> None:
    m, b = fit_closed_form([1, 2, 3, 4], [35000, 40000, 45000, 50000])
    assert m == pytest.approx(5000)
    assert b == pytest.approx(30000)


def test_closed_form_rejects_vertical_data() -> None:
    with pytest.raises(ValueError, match="identical"):
        fit_closed_form([2, 2, 2], [1, 2, 3])
