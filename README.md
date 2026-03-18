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
git clone https://github.com/amospagin/Py-semla.git
cd Py-semla
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

### Modifiers

```python
# Fix a loading to a specific value
"f1 =~ 1*x1 + x2 + x3"

# Label parameters
"f1 =~ a*x1 + b*x2 + c*x3"

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
| `cfa(model, data)` | Confirmatory Factor Analysis (auto-adds covariances between latent variables) |
| `sem(model, data)` | Structural Equation Model (no auto covariances) |
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

## Fit Indices

| Index | Description | Good Fit |
|-------|-------------|----------|
| Chi-square | Model test statistic | p > .05 |
| CFI | Comparative Fit Index | > .95 |
| TLI | Tucker-Lewis Index | > .95 |
| RMSEA | Root Mean Square Error of Approximation | < .06 |
| SRMR | Standardized Root Mean Square Residual | < .08 |

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

## Dependencies

- [NumPy](https://numpy.org/) >= 1.22
- [SciPy](https://scipy.org/) >= 1.8
- [pandas](https://pandas.pydata.org/) >= 1.4

## Roadmap

- [ ] DWLS/WLS estimators for ordinal data
- [ ] Standardized solutions (std.all, std.lv)
- [ ] Modification indices
- [ ] Multi-group analysis and measurement invariance
- [ ] Mean structure / intercepts
- [ ] Robust standard errors (MLR)

## License

[MIT](LICENSE)
