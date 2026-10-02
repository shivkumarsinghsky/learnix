# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-10-02

### Added
- `learnix` package: `LinearRegression` (gradient descent, optional feature standardisation, early stopping, loss history), `fit_closed_form`, `mean_squared_error`, `r2_score`, `residuals`, `load_csv_xy`.
- `data/salary.csv`: the 50-row dataset from the notebook, so lessons and tests share it.
- Lesson 5 trains on the full dataset and compares gradient descent with the exact solution.
- 30 tests: gradient checks, convergence, scikit-learn parity, divergence handling, save/load, and that every lesson runs.
- README with the maths, a training-loop diagram and results; ADRs; CI and security workflows; MIT license.

### Changed
- Lessons moved from `src/` to `lessons/` and numbered as a learning path; the notebook is now lesson 6.
- Models are saved as JSON instead of pickle (ADR-002).
- The default learning rate is 0.05 instead of 0.000001, which did not converge.
- `requirements.txt` renamed to `requirements-notebooks.txt`; the core package has no dependencies.

### Fixed
- Lesson 1's `predictSalary` ignored its argument and read a global variable.
- Training with too large a learning rate now raises `TrainingDivergedError` instead of producing NaN.

### Removed
- Committed build artefacts: `.DS_Store`, `__pycache__`, and two trained `.pkl` model files.

## [0.1.0] - 2026-08-05

### Added
- Linear regression lessons: prediction, residuals, MSE and gradient descent; pandas/scikit-learn notebook.
