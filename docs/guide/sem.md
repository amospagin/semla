# Structural Equation Modeling

SEM combines a measurement model (CFA) with structural paths (regressions between latent variables).

## Basic SEM

```python
from semla import sem

model = """
    # Measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8

    # Structural model (regressions)
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
"""

fit = sem(model, data=df)
fit.summary()
```

## Difference Between `cfa()` and `sem()`

| | `cfa()` | `sem()` |
|---|---------|---------|
| Latent covariances | Auto-added (all factors correlate) | Only if specified |
| Use case | Testing measurement structure | Testing causal/structural paths |

If your model has regressions between latent variables (`~`), use `sem()`. If it's purely a factor model, use `cfa()`.

## Observed Variable Regressions

You can also regress observed variables:

```python
model = """
    f1 =~ x1 + x2 + x3
    y ~ f1 + age + gender
"""
fit = sem(model, data=df)
```

## Correlated Residuals

Sometimes indicators share variance beyond their factor (e.g., similar wording):

```python
model = """
    f1 =~ x1 + x2 + x3 + x4
    x1 ~~ x2   # correlated residuals
"""
```

!!! note
    Adding correlated residuals should be theoretically motivated, not just driven by modification indices.
