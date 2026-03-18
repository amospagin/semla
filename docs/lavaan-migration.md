# Coming from lavaan?

If you know lavaan in R, you already know semla. The syntax is identical and the API is deliberately similar.

## Side-by-Side Comparison

=== "lavaan (R)"

    ```r
    library(lavaan)

    model <- '
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
    '

    fit <- cfa(model, data = HolzingerSwineford1939)
    summary(fit, fit.measures = TRUE, standardized = TRUE)
    modindices(fit, sort = TRUE)
    ```

=== "semla (Python)"

    ```python
    from semla import cfa
    from semla.datasets import HolzingerSwineford1939

    model = """
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
    """

    df = HolzingerSwineford1939()
    fit = cfa(model, data=df)
    fit.summary()
    fit.standardized_estimates()
    fit.modindices(sort=True)
    ```

## Function Mapping

| lavaan (R) | semla (Python) |
|------------|----------------|
| `cfa(model, data)` | `cfa(model, data)` |
| `sem(model, data)` | `sem(model, data)` |
| `summary(fit)` | `fit.summary()` |
| `fitMeasures(fit)` | `fit.fit_indices()` |
| `parameterEstimates(fit)` | `fit.estimates()` |
| `standardizedSolution(fit)` | `fit.standardized_estimates()` |
| `modindices(fit)` | `fit.modindices()` |
| `cfa(model, data, group="x")` | `cfa(model, data, group="x")` |
| `cfa(model, data, ordered=TRUE)` | `cfa(model, data, estimator="DWLS")` |
| `anova(fit1, fit2)` | `chi_square_diff_test(fit1, fit2)` |

## Key Differences

1. **Return values** — lavaan uses R's S4 objects with `summary()`, `coef()`, etc. semla returns objects with methods like `.summary()`, `.estimates()`.

2. **DataFrames** — semla returns pandas DataFrames instead of R data.frames. You can filter, sort, and manipulate them with standard pandas operations.

3. **Ordinal data** — lavaan uses `ordered=TRUE`, semla uses `estimator="DWLS"`.

4. **Output format** — semla's summary is modeled after lavaan's but not identical. The same information is there, just formatted slightly differently.

## What's Not Yet Supported

Features in lavaan that semla doesn't have yet:

- Mean structure / intercepts (`~1` — parsed but not estimated)
- Equality constraints (`a*x1` labels parsed but not enforced)
- Defined parameters (`:=` for indirect effects)
- `MLR` estimator (robust ML)
- `bootstrapLavaan()` for bootstrap CIs
- `lavPredict()` for factor scores
- `reliabilityL2()` for reliability

See the [Roadmap](roadmap.md) for planned additions.
