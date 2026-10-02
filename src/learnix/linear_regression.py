"""Simple linear regression y = m*x + b, trained with batch gradient descent in pure Python.

Every step is written out explicitly (no NumPy) so the maths in the README maps line by line
to the code: predict -> residuals -> MSE loss -> gradients -> parameter update.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from learnix._validation import check_xy

FORMAT = "learnix.LinearRegression"
FORMAT_VERSION = 1


class TrainingDivergedError(RuntimeError):
    """Raised when the loss becomes infinite or NaN (usually: learning rate too high)."""


@dataclass
class LinearRegression:
    """One-feature linear regression trained by batch gradient descent.

    learning_rate: step size for each update.
    epochs: maximum number of passes over the data.
    standardize: train on (x - mean) / std and convert the result back. This keeps
        gradient descent stable when x has a large range, without changing the final line.
    tolerance: stop early once the loss improves by less than this between epochs
        (0 disables early stopping).
    """

    learning_rate: float = 0.05
    epochs: int = 5000
    standardize: bool = False
    tolerance: float = 0.0
    m: float = 0.0
    b: float = 0.0
    loss_history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.tolerance < 0:
            raise ValueError("tolerance must not be negative")

    # --- the model -------------------------------------------------------------------

    def predict(self, xs: Sequence[float]) -> list[float]:
        """y_hat = m * x + b for every x."""
        return [self.m * x + self.b for x in xs]

    @staticmethod
    def gradients(
        xs: Sequence[float], ys: Sequence[float], m: float, b: float
    ) -> tuple[float, float]:
        """Partial derivatives of MSE with respect to m and b.

        dL/dm = -(2/n) * sum(x_i * (y_i - y_hat_i))
        dL/db = -(2/n) * sum(y_i - y_hat_i)
        """
        n = len(xs)
        errors = [y - (m * x + b) for x, y in zip(xs, ys, strict=True)]
        grad_m = (-2 / n) * sum(x * e for x, e in zip(xs, errors, strict=True))
        grad_b = (-2 / n) * sum(errors)
        return grad_m, grad_b

    @staticmethod
    def loss(xs: Sequence[float], ys: Sequence[float], m: float, b: float) -> float:
        """Mean squared error of the line (m, b) on the data."""
        return sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys, strict=True)) / len(xs)

    # --- training --------------------------------------------------------------------

    def fit(
        self, xs: Sequence[float], ys: Sequence[float], *, verbose: bool = False
    ) -> LinearRegression:
        """Learn m and b from the data. Returns self so calls can be chained."""
        check_xy(xs, ys)
        if self.standardize:
            mean_x = sum(xs) / len(xs)
            std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
            if std_x == 0:
                raise ValueError("cannot standardize: all x values are identical")
            train_xs = [(x - mean_x) / std_x for x in xs]
        else:
            mean_x, std_x = 0.0, 1.0
            train_xs = list(xs)

        m, b = 0.0, 0.0
        self.loss_history = []
        report_every = max(1, self.epochs // 10)
        for epoch in range(1, self.epochs + 1):
            try:
                current = self.loss(train_xs, ys, m, b)
            except OverflowError:
                current = math.inf
            if not math.isfinite(current):
                raise TrainingDivergedError(
                    f"loss became {current} at epoch {epoch}; lower learning_rate "
                    "or use standardize=True"
                )
            self.loss_history.append(current)
            grad_m, grad_b = self.gradients(train_xs, ys, m, b)
            m -= self.learning_rate * grad_m
            b -= self.learning_rate * grad_b
            if verbose and epoch % report_every == 0:
                print(f"epoch {epoch:>6}  loss {current:,.2f}")
            if (
                self.tolerance
                and len(self.loss_history) > 1
                and abs(self.loss_history[-2] - current) < self.tolerance
            ):
                break

        # Undo the scaling:
        #   m_z * (x - mean) / std + b_z  ==  (m_z / std) * x + (b_z - m_z * mean / std)
        self.m = m / std_x
        self.b = b - m * mean_x / std_x
        return self

    # --- persistence -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "m": self.m,
            "b": self.b,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "standardize": self.standardize,
                "tolerance": self.tolerance,
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write the learned parameters as JSON (data only, safe to load)."""
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LinearRegression:
        """Read a model written by save(). Rejects files that are not learnix models."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version {data.get('format_version')!r}")
        params = data.get("hyperparameters", {})
        model = cls(
            learning_rate=float(params.get("learning_rate", 0.05)),
            epochs=int(params.get("epochs", 5000)),
            standardize=bool(params.get("standardize", False)),
            tolerance=float(params.get("tolerance", 0.0)),
        )
        model.m = float(data["m"])
        model.b = float(data["b"])
        return model
