# Latent Growth Curve Models

Growth curve models estimate trajectories of change over time using latent intercept and slope factors.

## Linear Growth Model

```python
from semla import growth

model = """
    i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
    s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4
"""

fit = growth(model, data=df)
fit.summary()
```

- **`i`** (intercept): starting level — all loadings fixed to 1
- **`s`** (slope): rate of change — loadings fixed to time scores (0, 1, 2, 3)
- `growth()` automatically estimates latent means and fixes observed intercepts to 0

## Key Output

```python
# Latent means (intercept mean = average starting level, slope mean = average change)
fit.estimates()  # look for i ~1 and s ~1 rows

# Latent variances (individual differences in starting level and rate of change)
# Intercept-slope covariance (do people who start higher change faster?)
```

## Nonlinear (Free-Loading) Growth

Free one or more time loadings to estimate nonlinear trajectories:

```python
model = """
    i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
    s =~ 0*y1 + 1*y2 + NA*y3 + 3*y4
"""

fit = growth(model, data=df)
```

`NA*y3` frees the loading for y3, letting the data estimate the shape of change between time points 2 and 4. The first (0) and last (3) loadings remain fixed to anchor the scale.

## Piecewise Growth

Model change with different rates across time periods:

```python
model = """
    i  =~ 1*y1 + 1*y2 + 1*y3 + 1*y4 + 1*y5
    s1 =~ 0*y1 + 1*y2 + 2*y3 + 2*y4 + 2*y5
    s2 =~ 0*y1 + 0*y2 + 0*y3 + 1*y4 + 2*y5
"""

fit = growth(model, data=df)
```

- `s1`: slope for the first phase (y1–y3)
- `s2`: slope for the second phase (y3–y5)

## Tips

- Time scores in the slope loadings define the time metric. Use `0, 1, 2, 3` for equally spaced, or `0, 0.5, 1, 3` for unequal intervals
- The intercept-slope covariance tells you whether initial status predicts change
- With only 3 time points, only linear growth is identified (no free loadings)
- Add covariates by regressing `i` and `s` on observed predictors: `i ~ age; s ~ age`
