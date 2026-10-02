# ADR-002: JSON model files instead of pickle

## Context
The first version saved the trained model with `pickle`. Loading a pickle file can execute
arbitrary code, so a model file shared by someone else is a security risk. Pickles are also tied
to the class definition at the time they were written, and two generated `.pkl` files had been
committed to the repository.

## Decision
`LinearRegression.save()` writes a small JSON document: a format marker, a format version, the
learned parameters and the hyperparameters. `load()` rejects files without the marker or with an
unknown version. Trained model files are git-ignored.

## Alternatives
- **pickle / joblib:** convenient for arbitrary objects; unsafe for untrusted files.
- **ONNX or similar:** standard for deployment, but far heavier than two numbers need.

## Trade-offs
Every new model type needs its own `to_dict()`/`load()` code.

## Consequences
Model files are human-readable, diff-able and safe to load. The scikit-learn notebook still uses
joblib, which is that library's convention, and its output is git-ignored too.
