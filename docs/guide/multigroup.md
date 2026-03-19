# Multi-Group Analysis

Multi-group CFA tests whether a measurement model holds across different groups (e.g., gender, schools, countries). This is called **measurement invariance** testing.

## Invariance Levels

| Level | What's Equal | Tests |
|-------|-------------|-------|
| Configural | Model structure only | Same factors in all groups |
| Metric | + Factor loadings | Factors have same meaning |
| Scalar | + Intercepts | Group means are comparable |
| Strict | + Residual variances | Same measurement precision |

## Configural Invariance

Same model structure across groups, all parameters free to vary:

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit_config = cfa(model, data=df, group="school")
fit_config.summary()
```

## Metric Invariance

Factor loadings constrained equal across groups:

```python
fit_metric = cfa(model, data=df, group="school", invariance="metric")
```

## Scalar Invariance

Loadings and intercepts constrained equal (requires mean structure):

```python
fit_scalar = cfa(model, data=df, group="school", invariance="scalar")
```

## Strict Invariance

Loadings, intercepts, and residual variances constrained equal:

```python
fit_strict = cfa(model, data=df, group="school", invariance="strict")
```

## Testing Invariance

Compare nested models with the chi-square difference test:

```python
from semla import chi_square_diff_test

diff = chi_square_diff_test(fit_metric, fit_config)
print(diff)
# {'chi_sq_diff': 8.14, 'df_diff': 6, 'p_value': 0.228}
```

If **p > .05**, the more constrained model fits equally well, supporting that level of invariance.

## Automated Testing (Recommended)

Use `measurementInvariance()` to run the full hierarchy in one call:

```python
from semla import measurementInvariance

result = measurementInvariance(model, data=df, group="school")

result.summary()        # formatted table with PASS/FAIL decisions
result.table()          # DataFrame with all test statistics
result.highest_level    # e.g., "metric"
result["metric"]        # access individual fit objects
```

Output:
```
Level              χ²    df     CFI   RMSEA      Δχ²   Δdf   p(Δχ²)   Decision
------------------------------------------------------------------------
configural    115.084    48   0.924   0.068                           baseline
metric        123.222    54   0.921   0.065    8.138     6    0.228       PASS
scalar        202.746    63   0.841   0.086   79.524     9    0.000       FAIL
strict        219.334    72   0.832   0.083   16.588     9    0.056       PASS
```

Decision criteria: **PASS** if Δχ² p > .05 AND ΔCFI < .01.

## Manual Sequential Testing

You can also test each step individually:

```python
fit_config = cfa(model, data=df, group="school", invariance="configural")
fit_metric = cfa(model, data=df, group="school", invariance="metric")
fit_scalar = cfa(model, data=df, group="school", invariance="scalar")
fit_strict = cfa(model, data=df, group="school", invariance="strict")

# Test each step
print("Metric vs Configural:", chi_square_diff_test(fit_metric, fit_config))
print("Scalar vs Metric:",    chi_square_diff_test(fit_scalar, fit_metric))
print("Strict vs Scalar:",    chi_square_diff_test(fit_strict, fit_scalar))
```

## Per-Group Estimates

```python
est = fit_config.estimates()
est[est["group"] == "Pasteur"]
est[est["group"] == "Grant-White"]
```
