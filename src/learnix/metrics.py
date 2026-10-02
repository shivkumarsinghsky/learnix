"""Regression error metrics, written out step by step."""

from __future__ import annotations

from collections.abc import Sequence

from learnix._validation import check_xy


def residuals(actual: Sequence[float], predicted: Sequence[float]) -> list[float]:
    """Residual for each sample: actual - predicted."""
    check_xy(actual, predicted)
    return [a - p for a, p in zip(actual, predicted, strict=True)]


def mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """MSE = (1/n) * sum((actual - predicted)^2)."""
    errors = residuals(actual, predicted)
    return sum(e * e for e in errors) / len(errors)


def r2_score(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot.

    1.0 is a perfect fit; 0.0 is no better than always predicting the mean.
    """
    errors = residuals(actual, predicted)
    mean = sum(actual) / len(actual)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((a - mean) ** 2 for a in actual)
    if ss_tot == 0:
        raise ValueError("r2_score is undefined when every actual value is the same")
    return 1 - ss_res / ss_tot
