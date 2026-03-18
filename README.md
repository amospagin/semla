<p align="center">
  <h1 align="center">semla</h1>
  <p align="center">
    <strong>Structural Equation Modeling with lavaan-style syntax for Python</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://pypi.org/project/semla/"><img src="https://img.shields.io/badge/version-0.1.0-orange.svg" alt="Version"></a>
  </p>
</p>

---

> **Note:** semla is in early development (v0.1.0). The API may change, and results should be validated against established tools like lavaan before use in published research. Contributions and bug reports are welcome!

**semla** brings the familiar [lavaan](https://lavaan.ugent.be/) model syntax from R to Python. If you know lavaan, you already know semla.

Specify CFA and SEM models with the same `=~`, `~`, and `~~` operators — and get publication-ready fit indices, parameter estimates, and standard errors.

## Installation

```bash
pip install semla
```

Or install from source:

```bash
git clone https://github.com/amospagin/semla.git
cd semla
pip install -e .
```

## Quick Start

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

# Load the classic dataset (301 students, 9 mental ability tests)
df = HolzingerSwineford1939()

# Define a 3-factor CFA model — same syntax as lavaan
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Fit the model
fit = cfa(model, data=df)

# View lavaan-style summary
fit.summary()
```

**Output:**

```
semla 0.1.0 — SEM Results
============================================================

  Estimator                                           ML
  Number of observations                            301

Model Test User Model:

  Test statistic                                 85.022
  Degrees of freedom                                 24
  P-value (Chi-square)                            0.000

Fit Indices:

  CFI                                             0.931
  TLI (NNFI)                                      0.896
  RMSEA                                           0.092
  90% CI RMSEA                       [ 0.000,  0.114]
  SRMR                                            0.065

Parameter Estimates:

  Latent Variables:
    visual =~
      x1                 1.000                               (fixed)
      x2                 0.554     0.109     5.058     0.000
      x3                 0.729     0.117     6.209     0.000
    textual =~
      x4                 1.000                               (fixed)
      x5                 1.113     0.065    17.099     0.000
      x6                 0.926     0.056    16.454     0.000
    speed =~
      x7                 1.000                               (fixed)
      x8                 1.180     0.151     7.838     0.000
      x9                 1.082     0.195     5.534     0.000
  ...

============================================================
```

## Working with Results

```python
# Fit indices as a dictionary
fit.fit_indices()
# {'chi_square': 85.3, 'df': 24, 'cfi': 0.931, 'tli': 0.896, 'rmsea': 0.092, ...}

# Parameter estimates as a pandas DataFrame
est = fit.estimates()
est[est["op"] == "=~"]  # just the factor loadings
```

## Model Syntax

semla uses the same operators as lavaan:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Latent variable definition | `visual =~ x1 + x2 + x3` |
| `~` | Regression | `dep ~ ind1 + ind2` |
| `~~` | (Co)variance | `x1 ~~ x2` |
| `~1` | Intercept | `y ~1` |
| `:=` | Defined parameter | `indirect := a*b` |

### Modifiers

```python
# Fix a loading to a specific value
"f1 =~ 1*x1 + x2 + x3"

# Equality constraints — same label = forced equal
"f1 =~ x1 + a*x2 + a*x3"  # x2 and x3 loadings constrained equal

# Free a normally-fixed parameter
"f1 =~ NA*x1 + x2 + x3"
```

### Multiple lines or semicolons

```python
# Newlines
model = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6
"""

# Semicolons
model = "f1 =~ x1 + x2 + x3; f2 =~ x4 + x5 + x6"

# Comments
model = """
    f1 =~ x1 + x2 + x3  # visual factor
    f2 =~ x4 + x5 + x6  # textual factor
"""
```

## Functions

| Function | Description |
|----------|-------------|
| `cfa(model, data)` | Confirmatory Factor Analysis (auto-adds latent covariances) |
| `sem(model, data)` | Structural Equation Model (no auto covariances) |
| `cfa(model, data, group="col")` | Multi-group CFA with measurement invariance testing |
| `cfa(model, data, estimator="DWLS")` | CFA for ordinal data using polychoric correlations |
| `cfa(model, data, estimator="bayes")` | Bayesian estimation via NumPyro NUTS sampler |
| `chi_square_diff_test(m1, m2)` | Nested model comparison via chi-square difference test |
| `Model(model, data)` | Direct model constructor |

### SEM Example

```python
from semla import sem

model = """
    # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8

    # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
"""

fit = sem(model, data=df)
fit.summary()
```

### Mediation Example

```python
from semla import sem

model = """
    M ~ a*X          # X -> M (path a)
    Y ~ b*M + c*X    # M -> Y (path b), X -> Y (direct, path c)

    indirect := a*b   # indirect effect
    total := a*b + c  # total effect
"""

fit = sem(model, data=df)
fit.defined_estimates()  # indirect effect with delta method SE
```

### Bayesian Estimation

semla supports full Bayesian SEM estimation via [NumPyro](https://num.pyro.ai/). Pass `estimator="bayes"` to use NUTS (No-U-Turn Sampler) with adaptive convergence monitoring.

```bash
pip install semla[bayes]  # installs numpyro and jax
```

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Bayesian estimation with data-adaptive priors (default)
fit = cfa(model, data=df, estimator="bayes")
fit.summary()

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

**Bayesian output:**

```
semla 0.1.0 — Bayesian SEM Results (NumPyro)
=================================================================

  Estimator                                           Bayes
  Chains                                                  4
  Draws per chain                                      2000
  Warmup per chain                                     1000
  Divergences                                             0

Parameter Estimates (posterior):

  lhs        op   rhs            mean   median       sd ci.lower ci.upper   rhat     ess
  -----------------------------------------------------------------------------------

  Latent Variables:
  visual     =~   x2            0.579    0.573    0.112    0.377    0.820  1.000    7224
  visual     =~   x3            0.762    0.753    0.122    0.552    1.026  1.001    5970
  ...
```

**Bayesian-specific methods:**

```python
fit.results.draws()         # raw posterior samples (DataFrame)
fit.results.estimates()     # mean, median, sd, CI, R-hat, ESS
fit.results.diagnostics()   # divergences, min ESS, max R-hat
fit.results.waic()          # WAIC model comparison
fit.results.loo()           # LOO-CV via PSIS
```

**Features:**
- Data-adaptive priors scaled by observed SDs (brms-style), or weak informative preset
- Adaptive convergence: auto-extends draws or increases adapt_delta when R-hat > 1.01
- Positive loading constraints prevent sign-flipping in structural models
- Parallel chain execution on CPU (one core per chain by default)
- WAIC and PSIS-LOO for Bayesian model comparison

## Fit Indices

| Index | Description | Good Fit |
|-------|-------------|----------|
| Chi-square | Model test statistic | p > .05 |
| CFI | Comparative Fit Index | > .95 |
| TLI | Tucker-Lewis Index | > .95 |
| RMSEA | Root Mean Square Error of Approximation | < .06 |
| SRMR | Standardized Root Mean Square Residual | < .08 |
| AIC | Akaike Information Criterion | Lower is better |
| BIC | Bayesian Information Criterion | Lower is better |

## Built-in Datasets

```python
from semla.datasets import HolzingerSwineford1939

df = HolzingerSwineford1939()
# 301 rows x 15 columns
# Variables: id, sex, ageyr, agemo, school, grade, x1-x9
```

**Holzinger & Swineford (1939):** Mental ability test scores for 301 seventh- and eighth-grade students from two schools. The 9 test variables load on three factors:

- **Visual:** x1 (visual perception), x2 (cubes), x3 (lozenges)
- **Textual:** x4 (paragraph comprehension), x5 (sentence completion), x6 (word meaning)
- **Speed:** x7 (speeded addition), x8 (speeded counting), x9 (speeded discrimination)

## Coming from lavaan?

| lavaan (R) | semla (Python) |
|------------|----------------|
| `library(lavaan)` | `from semla import cfa, sem` |
| `fit <- cfa(model, data=df)` | `fit = cfa(model, data=df)` |
| `summary(fit, fit.measures=TRUE)` | `fit.summary()` |
| `fitMeasures(fit)` | `fit.fit_indices()` |
| `parameterEstimates(fit)` | `fit.estimates()` |
| `standardizedSolution(fit)` | `fit.standardized_estimates()` |
| `modindices(fit)` | `fit.modindices()` |
| `cfa(model, data, group="x")` | `cfa(model, data, group="x")` |
| `cfa(model, data, ordered=TRUE)` | `cfa(model, data, estimator="DWLS")` |
| `blavaan::bcfa(model, data)` | `cfa(model, data, estimator="bayes")` |
| `summary(fit, rsquare=TRUE)` | `fit.r_squared()` |
| `parameterEstimates(fit)` (with `:=`) | `fit.defined_estimates()` |
| `lavPredict(fit)` | `fit.predict()` |
| `reliability(fit)` | `fit.reliability()` |
| `bootstrapLavaan(fit)` | `fit.bootstrap(nboot=1000)` |

## Dependencies

**Core:**
- [NumPy](https://numpy.org/) >= 1.22
- [SciPy](https://scipy.org/) >= 1.8
- [pandas](https://pandas.pydata.org/) >= 1.4

**Bayesian estimation** (optional — `pip install semla[bayes]`):
- [NumPyro](https://num.pyro.ai/) >= 0.13
- [JAX](https://jax.readthedocs.io/) >= 0.4

## Roadmap

**v0.1.0 (current):**
- [x] ML estimation with lavaan-validated results
- [x] DWLS estimator for ordinal data (polychoric correlations)
- [x] Standardized solutions (std.all, std.lv)
- [x] Modification indices
- [x] Multi-group CFA (configural, metric, scalar, strict invariance)
- [x] Chi-square difference test
- [x] AIC / BIC / adjusted BIC information criteria
- [x] R-squared for endogenous variables
- [x] Mean structure and intercepts (~1)
- [x] Equality constraints via parameter labels (a*x1 + a*x2)
- [x] Indirect effects and mediation (:= operator with delta method SEs)
- [x] Reliability measures — McDonald's omega and Cronbach's alpha
- [x] Factor score prediction (regression and Bartlett methods)
- [x] Bootstrap confidence intervals
- [x] Robust ML estimator (MLR) with Satorra-Bentler scaled chi-square
- [x] FIML for missing data
- [x] IRT models (1PL, 2PL, GRM) with ICC, information functions, and ability estimation
- [x] Residual diagnostics and Mardia's multivariate normality test
- [x] Input validation and Heywood case warnings
- [x] Bayesian MCMC estimation via NumPyro (NUTS sampler, adaptive priors, WAIC/LOO)


## License

[MIT](LICENSE)
