"""Shared input checks, so every function fails with the same clear message."""

from __future__ import annotations

import math
from collections.abc import Sequence


def check_xy(xs: Sequence[float], ys: Sequence[float]) -> None:
    """Raise ValueError unless xs and ys are non-empty, equal-length and finite."""
    if len(xs) == 0:
        raise ValueError("need at least one sample")
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have the same length ({len(xs)} != {len(ys)})")
    if not all(math.isfinite(v) for v in (*xs, *ys)):
        raise ValueError("inputs must be finite numbers (no NaN or infinity)")
