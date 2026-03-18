# API Reference

## Main Functions

### `cfa(model, data, group=None, estimator="ML", **kwargs)`

Fit a Confirmatory Factor Analysis model. Automatically adds covariances between latent variables.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model syntax in lavaan format |
| `data` | `DataFrame` | Data with columns matching observed variables |
| `group` | `str`, optional | Column name for multi-group analysis |
| `estimator` | `str` | `"ML"` (default) or `"DWLS"` for ordinal data |
| `invariance` | `str` | `"configural"` or `"metric"` (multi-group only) |

**Returns:** `Model` or `MultiGroupModel`

---

### `sem(model, data, group=None, estimator="ML", **kwargs)`

Fit a Structural Equation Model. Same as `cfa()` but does NOT auto-add covariances between latent variables.

---

### `chi_square_diff_test(model_restricted, model_free)`

Chi-square difference test for nested model comparison.

**Returns:** `dict` with `chi_sq_diff`, `df_diff`, `p_value`

---

## Model Object

Returned by `cfa()` and `sem()` for single-group models.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Print lavaan-style summary |
| `fit_indices()` | `dict` | Chi-square, CFI, TLI, RMSEA, SRMR |
| `estimates()` | `DataFrame` | Parameter estimates with SEs, z-values, p-values |
| `standardized_estimates(type)` | `DataFrame` | `"std.all"` or `"std.lv"` standardization |
| `modindices(min_mi=0)` | `DataFrame` | Modification indices sorted by MI |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `converged` | `bool` | Whether the optimizer converged |

---

## MultiGroupModel Object

Returned by `cfa()` and `sem()` when `group=` is specified.

### Methods

Same as `Model`, except `estimates()` includes a `group` column.

---

## Datasets

### `semla.datasets.HolzingerSwineford1939()`

Classic dataset: 301 students, 9 mental ability tests, 2 schools.

**Columns:** id, sex, ageyr, agemo, school, grade, x1–x9

**Factors:** visual (x1–x3), textual (x4–x6), speed (x7–x9)
