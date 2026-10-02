# ADR-003: Verify training against exact solutions and scikit-learn

## Context
A gradient-descent implementation can look plausible and still be wrong: a sign error in a
gradient, a learning rate that silently fails to converge, or a scaling bug. The first version
used a learning rate of 0.000001, which moved the slope from 0 to about 22 when the right answer
was 5000, and nothing flagged it.

## Decision
The tests check the implementation against three independent references:

1. Analytic gradients against numerical central differences.
2. Gradient descent against the closed-form least-squares solution.
3. The trained model against scikit-learn's `LinearRegression` (skipped if scikit-learn is not installed).

Training raises `TrainingDivergedError` if the loss becomes infinite or NaN, instead of returning
NaN parameters.

## Alternatives
- **Only check that the loss decreases:** catches divergence but not a wrong minimum.
- **Snapshot printed output:** brittle, and does not prove correctness.

## Trade-offs
scikit-learn is a development dependency.

## Consequences
Every later model (multiple regression, logistic regression) will ship with the same three kinds
of check.
