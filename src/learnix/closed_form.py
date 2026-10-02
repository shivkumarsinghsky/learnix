"""Ordinary least squares solved exactly, used to check gradient descent."""

from __future__ import annotations

from collections.abc import Sequence

from learnix._validation import check_xy


def fit_closed_form(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return (m, b) minimising MSE for y = m*x + b.

    m = cov(x, y) / var(x),  b = mean(y) - m * mean(x)
    """
    check_xy(xs, ys)
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        raise ValueError("all x values are identical, so the slope is undefined")
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    m = cov_xy / var_x
    return m, mean_y - m * mean_x
