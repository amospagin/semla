# Cookbook

Copy-paste recipes for common model patterns.

## Bifactor Model

A general factor plus specific group factors:

```python
from semla import cfa

model = """
    # General factor loads on all items
    g =~ x1 + x2 + x3 + x4 + x5 + x6

    # Specific factors (orthogonal to g)
    verbal  =~ x1 + x2 + x3
    quant   =~ x4 + x5 + x6

    # Fix specific-general covariances to 0
    g ~~ 0*verbal
    g ~~ 0*quant
    verbal ~~ 0*quant
"""

fit = cfa(model, data=df)
```

## MIMIC Model

Covariates predicting a latent variable and (optionally) direct effects on indicators:

```python
from semla import sem

model = """
    # Measurement model
    depression =~ d1 + d2 + d3 + d4

    # Covariates predict latent
    depression ~ age + gender

    # Direct effect on an indicator (differential item functioning)
    d2 ~ gender
"""

fit = sem(model, data=df)
```

## Multiple Mediators (Parallel)

```python
from semla import sem

model = """
    # Paths
    m1 ~ a1*X
    m2 ~ a2*X
    Y  ~ b1*m1 + b2*m2 + c*X

    # Indirect effects
    ind1  := a1*b1
    ind2  := a2*b2
    total := a1*b1 + a2*b2 + c
"""

fit = sem(model, data=df)
fit.defined_estimates()
```

## Serial Mediation

```python
model = """
    M1 ~ a*X
    M2 ~ d*M1 + X
    Y  ~ b*M2 + M1 + c*X

    indirect_serial := a*d*b
    indirect_total  := a*d*b + c
"""

fit = sem(model, data=df)
```

## Correlated Uniquenesses (Method Effects)

When indicators share method variance (e.g., same response format):

```python
model = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6

    # Items x1 and x4 share method (e.g., reverse-coded)
    x1 ~~ x4
    # Items x3 and x6 share method
    x3 ~~ x6
"""

fit = cfa(model, data=df)
```

## Orthogonal Factors

```python
model = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6
    f1 ~~ 0*f2
"""

fit = cfa(model, data=df)
```

## Second-Order Factor Model

```python
model = """
    # First-order factors
    verbal =~ v1 + v2 + v3
    quant  =~ q1 + q2 + q3
    memory =~ m1 + m2 + m3

    # Second-order factor
    g =~ verbal + quant + memory
"""

fit = sem(model, data=df)
```

## Constrained Factor Loadings

Tau-equivalent model (all loadings equal within a factor):

```python
model = """
    f =~ a*x1 + a*x2 + a*x3 + a*x4
"""

fit = cfa(model, data=df)
```

## What To Do When a Model Doesn't Converge

1. **Run diagnostics**: `fit.check()` for a summary of issues
2. **Check modification indices**: `fit.modindices(min_mi=5)` for missing paths
3. **Simplify**: start with a simpler model and add complexity
4. **Check data**: look for constant columns, extreme outliers, tiny sample
5. **Try different estimator**: `estimator="MLR"` is more robust to non-normality

## How To Handle Heywood Cases

A Heywood case (negative variance estimate) usually means:

- **Too few indicators**: each factor needs 3+ indicators
- **Missing path**: a cross-loading or correlated residual is needed
- **Small sample**: model is too complex for the data

```python
# Check what's wrong
fit.check()

# Look for missing paths
fit.modindices(min_mi=5)
```

## Choosing an Estimator

| Estimator | When to use |
|-----------|-------------|
| `ML` | Default. Continuous data, large sample, approximately normal |
| `MLR` | Continuous but non-normal data (robust SEs and scaled chi-square) |
| `DWLS` | Ordinal/categorical indicators (Likert scales) |
| `FIML` | Missing data (uses all available information) |
| `bayes` | Small samples, complex models, prior information |

```python
fit = cfa(model, data=df, estimator="MLR")
fit = cfa(model, data=df, estimator="DWLS")
fit = cfa(model, data=df, missing="fiml")
fit = cfa(model, data=df, estimator="bayes", chains=4, draws=2000)
```

## Reporting Checklist

Include in your paper:

- [ ] Model specification (path diagram or syntax)
- [ ] Sample size and estimator used
- [ ] Fit indices: chi-square (df, p), CFI, TLI, RMSEA (90% CI), SRMR
- [ ] All parameter estimates with SEs
- [ ] Standardized solution (`fit.standardized_estimates()`)
- [ ] For Bayesian: posterior summaries, R-hat, ESS, prior specification
