# Multi-Group Analysis

Multi-group CFA tests whether a measurement model holds across different groups (e.g., gender, schools, countries). This is called **measurement invariance** testing.

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

This tests: *Does the same factor structure hold in both schools?*

## Metric Invariance

Factor loadings constrained equal across groups:

```python
fit_metric = cfa(model, data=df, group="school", invariance="metric")
```

This tests: *Do the factors mean the same thing in both groups?* Equal loadings imply the same unit of measurement.

## Testing Invariance

Compare models with the chi-square difference test:

```python
from semla import chi_square_diff_test

diff = chi_square_diff_test(fit_metric, fit_config)
print(diff)
# {'chi_sq_diff': 8.14, 'df_diff': 6, 'p_value': 0.228}
```

If **p > .05**, the more constrained model (metric) fits equally well, supporting that level of invariance.

## Invariance Levels

| Level | What's Equal | Tests |
|-------|-------------|-------|
| Configural | Model structure only | Same factors in all groups |
| Metric | + Factor loadings | Factors have same meaning |
| Scalar | + Intercepts | Group means are comparable |
| Strict | + Residual variances | Same measurement precision |

!!! note
    semla currently supports configural and metric invariance. Scalar invariance requires mean structure (intercepts), which is planned for a future release.

## Per-Group Estimates

```python
est = fit_config.estimates()

# Filter by group
est[est["group"] == "Pasteur"]
est[est["group"] == "Grant-White"]
```

## Verifying Equal Loadings

Under metric invariance, loadings should be identical across groups:

```python
est = fit_metric.estimates()
loadings = est[(est["op"] == "=~") & (est["free"])]
for ind in loadings["rhs"].unique():
    vals = loadings[loadings["rhs"] == ind]
    print(f"{ind}: {vals['est'].values}")  # should be identical
```
