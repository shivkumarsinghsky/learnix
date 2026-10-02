# ADR-001: Pure Python core, no NumPy

## Context
The purpose of this repository is to understand how models learn. Vectorised NumPy code is
fast, but it hides the per-sample loop that the maths describes, and it adds an install step
before the first lesson can run.

## Decision
The `learnix` package uses only the Python standard library. Predictions, residuals, loss and
gradients are written as explicit sums over samples. Libraries (pandas, scikit-learn) appear
only in the notebook and in an optional comparison test.

## Alternatives
- **NumPy from the start:** shorter code, but the link between formula and code is less visible.
- **Two implementations (loop and vectorised):** useful later, but doubles the surface now.

## Trade-offs
Training is slow on large datasets. That is acceptable for datasets of tens to thousands of rows.

## Consequences
Lessons run on any Python 3.10+ install with no dependencies. A vectorised version can be added
as a later lesson and tested against this one.
