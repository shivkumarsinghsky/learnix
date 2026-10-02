"""learnix: machine learning fundamentals implemented from scratch in pure Python."""

from learnix.closed_form import fit_closed_form
from learnix.data import load_csv_xy
from learnix.linear_regression import LinearRegression, TrainingDivergedError
from learnix.metrics import mean_squared_error, r2_score, residuals

__all__ = [
    "LinearRegression",
    "TrainingDivergedError",
    "fit_closed_form",
    "load_csv_xy",
    "mean_squared_error",
    "r2_score",
    "residuals",
]

__version__ = "0.2.0"
