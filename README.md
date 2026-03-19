<p align="center">
  <h1 align="center">semla</h1>
  <p align="center">
    <strong>Latent variable modeling and SEM in Python, with lavaan syntax</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://pypi.org/project/semla/"><img src="https://img.shields.io/badge/version-0.1.0-orange.svg" alt="Version"></a>
  </p>
</p>

---

> **Early development (v0.1.0).** The API may change. Results should be validated against established tools before use in published research.

**semla** is a Python package for structural equation modeling, confirmatory factor analysis, latent growth curves, IRT, and other latent variable models. It uses [lavaan](https://lavaan.ugent.be/)-style syntax for model specification, so if you know lavaan, you already know the syntax.

Choose between frequentist estimation (ML, MLR, DWLS, FIML) and full Bayesian MCMC inference — from the same interface. Run batches of Bayesian models in parallel across CPU cores and GPU.

## Installation

```bash
pip install semla              # core (ML, MLR, DWLS, FIML, IRT, multi-group)
pip install semla[bayes]       # + Bayesian estimation (NumPyro/JAX, CPU)
pip install semla[bayes-cuda]  # + Bayesian estimation (NumPyro/JAX, NVIDIA GPU)
```

Or install from source:

```bash
git clone https://github.com/amospagin/semla.git
cd semla
pip install -e ".[bayes]"
```

## Quick Start

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit = cfa(model, data=df)
fit.summary()
```

## Features at a Glance

| Category | What's available |
|----------|-----------------|
| **Model types** | CFA, SEM with regressions, mediation (`:=` indirect effects), growth curves (linear and nonlinear), higher-order factor models, cross-lagged panel models, IRT (1PL, 2PL, GRM) |
| **Estimators** | ML, MLR (robust Satorra-Bentler), DWLS (ordinal/polychoric), FIML (missing data), Bayesian MCMC (NumPyro NUTS) |
| **Multi-group** | Configural, metric, scalar, strict invariance; automated `measurementInvariance()` testing |
| **Bayesian** | Adaptive priors, adaptive convergence, parallel chains, WAIC, PSIS-LOO, posterior draws |
| **Batch estimation** | Run many Bayesian models in parallel across CPU cores + GPU with `batch_bayes()` |
| **Diagnostics** | Fit indices (CFI, TLI, RMSEA, SRMR), modification indices, residuals, Mardia's normality test |
| **Constraints** | Equality constraints (labels), nonlinear inequality/equality constraints (`>`, `<`, `==`) |
| **Inference** | Standard errors, bootstrap CIs, chi-square difference test, model comparison table, R-squared, reliability (omega, alpha) |
| **Post-estimation** | Factor scores (regression, Bartlett), standardized solutions, model-implied matrices (`fitted()`), parameter covariance matrix (`vcov()`) |

## Model Syntax

semla uses the same operators as lavaan:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Latent variable definition | `visual =~ x1 + x2 + x3` |
| `~` | Regression | `dep ~ ind1 + ind2` |
| `~~` | (Co)variance | `x1 ~~ x2` |
| `~1` | Intercept | `y ~1` |
| `:=` | Defined parameter | `indirect := a*b` |
| `>` `<` `>=` `<=` | Inequality constraint | `a > 0` |
| `==` | Nonlinear equality constraint | `a*b == 0.5` |

### Modifiers

```python
"f1 =~ 1*x1 + x2 + x3"        # fix a loading to a specific value
"f1 =~ x1 + a*x2 + a*x3"      # equality constraints (same label = forced equal)
"f1 =~ NA*x1 + x2 + x3"       # free a normally-fixed parameter
```

## CFA and SEM

```python
from semla import cfa, sem

# CFA — auto-adds covariances between latent variables
fit = cfa("f1 =~ x1 + x2 + x3; f2 =~ x4 + x5 + x6", data=df)

# SEM — with structural regressions
fit = sem("""
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
""", data=df)

# Mediation with indirect effects
fit = sem("""
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
    total := a*b + c
""", data=df)
fit.defined_estimates()  # indirect effect with delta method SE
```

## Growth Curve Models

```python
from semla import growth

# Linear growth over 4 time points
fit = growth("""
    i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
    s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4
""", data=df)

# Nonlinear growth — free the y3 time loading
fit = growth("""
    i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
    s =~ 0*y1 + 1*y2 + NA*y3 + 3*y4
""", data=df)
```

## Bayesian Estimation

Switch to full Bayesian inference with `estimator="bayes"`. Uses [NumPyro](https://num.pyro.ai/) NUTS sampler with data-adaptive priors.

```python
# Default: data-adaptive priors, 4 parallel chains
fit = cfa(model, data=df, estimator="bayes")

# Customize sampling
fit = cfa(model, data=df, estimator="bayes",
          chains=4, warmup=1000, draws=2000, cores=4, seed=42)

# Weak informative priors
fit = cfa(model, data=df, estimator="bayes", priors="weak")

# Per-parameter or matrix-level prior overrides
from semla.priors import Normal, InverseGamma
fit = cfa(model, data=df, estimator="bayes",
          priors={"loadings": Normal(0, 1), "f1=~x2": Normal(0.7, 0.2)})
```

```python
fit.summary()                # Bayesian summary with posterior stats and R-hat
fit.results.draws()          # raw posterior samples as DataFrame
fit.results.estimates()      # mean, median, sd, CI, R-hat, ESS
fit.results.diagnostics()    # divergences, min ESS, max R-hat
fit.results.waic()           # WAIC model comparison
fit.results.loo()            # LOO-CV via PSIS
```

## Batch Bayesian Estimation

Run many Bayesian models in parallel using `batch_bayes()`. Models are distributed across CPU cores and (optionally) a GPU using a complexity-based scheduler.

```python
from semla import batch_bayes

models = {
    "2factor": "f1 =~ x1+x2+x3\nf2 =~ x4+x5+x6",
    "1factor": "f1 =~ x1+x2+x3+x4+x5+x6",
    "3factor": "f1 =~ x1+x2\nf2 =~ x3+x4\nf3 =~ x5+x6",
}

results = batch_bayes(models, data=df, cpu_cores=6, gpu="auto",
                      warmup=1000, draws=2000)

results.summary_table()      # status, backend, WAIC for all models
results.compare()            # rank models by WAIC
results["2factor"].summary   # individual model summary

# Per-model priors for sensitivity analysis
results = batch_bayes(models, data=df, priors={
    "2factor": "weak",
    "1factor": {"loadings": Normal(0, 1)},
})

# Explicit GPU assignment
results = batch_bayes(models, data=df,
                      gpu_models=["2factor", "1factor"])
```

## IRT Models

Fit Item Response Theory models using CFA parameterization. Supports binary (1PL, 2PL) and ordinal (GRM) items.

```python
from semla import irt

# 2PL model for binary items
fit = irt(items=["item1", "item2", "item3", "item4", "item5"],
          data=df, model_type="2PL")

fit.summary()
fit.irt_params()              # discrimination, difficulty per item
fit.icc()                     # item characteristic curves
fit.item_information()        # item information functions
fit.test_information()        # total test information and SE
fit.abilities()               # person ability (theta) estimates
```

| Model | Description |
|-------|-------------|
| `1PL` | Rasch model — equal discrimination, estimate difficulty |
| `2PL` | Two-parameter — estimate both discrimination and difficulty |
| `GRM` | Graded Response Model — for ordinal (Likert-scale) items |

## Multi-Group Analysis

Test measurement invariance across groups in one call:

```python
from semla import measurementInvariance

result = measurementInvariance(model, data=df, group="gender")
result.summary()         # formatted table with PASS/FAIL decisions
result.highest_level     # e.g., "metric"
result["metric"]         # access individual fit objects
```

Or fit each level manually:

```python
from semla import cfa, chi_square_diff_test, compare_models

fit_config = cfa(model, data=df, group="gender", invariance="configural")
fit_metric = cfa(model, data=df, group="gender", invariance="metric")

chi_square_diff_test(fit_metric, fit_config)
```

## Estimators

| Estimator | Usage | When to use |
|-----------|-------|-------------|
| `ML` (default) | `cfa(model, data=df)` | Continuous, complete data, multivariate normal |
| `MLR` | `cfa(model, data=df, estimator="MLR")` | Non-normal continuous data (Satorra-Bentler correction) |
| `DWLS` | `cfa(model, data=df, estimator="DWLS")` | Ordinal/categorical data (polychoric correlations) |
| `FIML` | `cfa(model, data=df, missing="fiml")` | Data with missing values |
| `Bayes` | `cfa(model, data=df, estimator="bayes")` | Bayesian inference, small samples, complex models |

## Working with Results

```python
# Fit indices
fit.fit_indices()
# {'chi_square': 85.3, 'df': 24, 'cfi': 0.931, 'rmsea': 0.092, 'srmr': 0.065, ...}

# Parameter estimates as DataFrame
fit.estimates()

# Standardized solution
fit.standardized_estimates(type="std.all")

# Model-implied covariance and mean matrices
fit.fitted()

# Parameter variance-covariance matrix
fit.vcov()

# Modification indices
fit.modindices(min_mi=5.0)

# Factor scores
fit.predict(method="regression")

# Bootstrap CIs
fit.bootstrap(nboot=1000, seed=42)

# Reliability
fit.reliability()  # McDonald's omega and Cronbach's alpha per factor

# R-squared for endogenous variables
fit.r_squared()

# Residual covariance matrix
fit.residuals(type="standardized")

# Multivariate normality check
from semla import mardia_test
mardia_test(df[["x1", "x2", "x3"]].values)
```

## Fit Indices

| Index | Description | Good Fit |
|-------|-------------|----------|
| Chi-square | Model test statistic | p > .05 |
| CFI | Comparative Fit Index | > .95 |
| TLI | Tucker-Lewis Index | > .95 |
| RMSEA | Root Mean Square Error of Approximation | < .06 |
| SRMR | Standardized Root Mean Square Residual | < .08 |
| AIC / BIC | Information criteria | Lower is better |
| WAIC | Widely Applicable IC (Bayesian) | Lower is better |
| LOO | Leave-One-Out CV (Bayesian) | Lower is better |

## Validation

semla is validated against [lavaan](https://lavaan.ugent.be/) 0.6-20. Parameter estimates, standard errors, and fit indices are compared across a range of model types:

- Simple and multi-factor CFA (with and without mean structure, equality constraints)
- SEM with regressions and mediation
- Higher-order (second-order) factor models
- Linear and nonlinear latent growth curves
- Cross-lagged panel models
- Multi-group invariance (configural through strict)
- MLR, DWLS, and FIML estimators

Estimates and SEs typically match lavaan within 0.01, and fit indices within 0.005. The full validation suite (200+ tests) runs against hardcoded lavaan reference values. See `tests/test_validate_*.py` for details.

## Coming from R?

| lavaan / blavaan / mirt (R) | semla (Python) |
|------------------------------|----------------|
| `library(lavaan)` | `from semla import cfa, sem` |
| `fit <- cfa(model, data=df)` | `fit = cfa(model, data=df)` |
| `fit <- growth(model, data=df)` | `fit = growth(model, data=df)` |
| `summary(fit, fit.measures=TRUE)` | `fit.summary()` |
| `fitMeasures(fit)` | `fit.fit_indices()` |
| `parameterEstimates(fit)` | `fit.estimates()` |
| `fitted(fit)` | `fit.fitted()` |
| `vcov(fit)` | `fit.vcov()` |
| `standardizedSolution(fit)` | `fit.standardized_estimates()` |
| `modindices(fit)` | `fit.modindices()` |
| `anova(fit1, fit2, fit3)` | `compare_models(m1=fit1, m2=fit2, m3=fit3)` |
| `cfa(model, data, group="x")` | `cfa(model, data, group="x")` |
| `measurementInvariance(model, data, group="x")` | `measurementInvariance(model, data, group="x")` |
| `cfa(model, data, ordered=TRUE)` | `cfa(model, data, estimator="DWLS")` |
| `blavaan::bcfa(model, data)` | `cfa(model, data, estimator="bayes")` |
| `mirt(data, 1, itemtype="2PL")` | `irt(items, data, model_type="2PL")` |
| `lavPredict(fit)` | `fit.predict()` |
| `reliability(fit)` | `fit.reliability()` |
| `bootstrapLavaan(fit)` | `fit.bootstrap(nboot=1000)` |

## Built-in Datasets

```python
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()  # 301 rows x 15 columns
```

**Holzinger & Swineford (1939):** Mental ability test scores for 301 seventh- and eighth-grade students. Nine tests loading on three factors: Visual (x1-x3), Textual (x4-x6), Speed (x7-x9).

## Dependencies

**Core:**
- [NumPy](https://numpy.org/) >= 1.22
- [SciPy](https://scipy.org/) >= 1.8
- [pandas](https://pandas.pydata.org/) >= 1.4

**Bayesian estimation** (optional — `pip install semla[bayes]`):
- [NumPyro](https://num.pyro.ai/) >= 0.13
- [JAX](https://jax.readthedocs.io/) >= 0.4

## License

[MIT](LICENSE)
