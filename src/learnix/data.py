"""Tiny CSV loader so lessons do not need pandas."""

from __future__ import annotations

import csv
from pathlib import Path


def load_csv_xy(path: str | Path, x_column: str, y_column: str) -> tuple[list[float], list[float]]:
    """Read two numeric columns from a CSV file with a header row."""
    xs: list[float] = []
    ys: list[float] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {x_column, y_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                xs.append(float(row[x_column]))
                ys.append(float(row[y_column]))
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: non-numeric value") from exc
    return xs, ys
