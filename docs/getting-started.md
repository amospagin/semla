# Getting Started

## Installation

```bash
pip install semla
```

## Your First CFA Model

A Confirmatory Factor Analysis tests whether observed variables load on hypothesized latent factors.

```python
from semla import cfa
from semla.datasets import HolzingerSwineford1939

# Load the classic dataset
df = HolzingerSwineford1939()

# Specify a 3-factor model
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Fit the model
fit = cfa(model, data=df)
```

## Examining Results

### Summary

```python
fit.summary()
```

This prints a lavaan-style summary with fit indices and parameter estimates.

### Fit Indices

```python
fit.fit_indices()
# {'chi_square': 85.0, 'df': 24, 'cfi': 0.931, 'tli': 0.896, ...}
```

| Index | Good Fit |
|-------|----------|
| CFI | > .95 |
| TLI | > .95 |
| RMSEA | < .06 |
| SRMR | < .08 |

### Parameter Estimates

```python
# Unstandardized
fit.estimates()

# Standardized (std.all)
fit.standardized_estimates()

# Just the loadings
est = fit.estimates()
est[est["op"] == "=~"]
```

### Modification Indices

Find where the model could be improved:

```python
fit.modindices(min_mi=5.0)
```

Large MI values suggest paths that, if freed, would significantly improve fit.

## Model Syntax

semla uses the same operators as lavaan:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Latent variable definition | `visual =~ x1 + x2 + x3` |
| `~` | Regression | `y ~ x1 + x2` |
| `~~` | (Co)variance | `x1 ~~ x2` |
| `~1` | Intercept | `y ~1` |

### Modifiers

```python
# Fix a loading
"f1 =~ 1*x1 + x2 + x3"

# Free the first loading (normally fixed to 1)
"f1 =~ NA*x1 + x2 + x3"
```

### Multiple lines or semicolons

```python
# Both work
model = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6
"""

model = "f1 =~ x1 + x2 + x3; f2 =~ x4 + x5 + x6"
```

## Next Steps

- [CFA Guide](guide/cfa.md) — deeper dive into CFA
- [SEM Guide](guide/sem.md) — structural models with regressions
- [Multi-Group Analysis](guide/multigroup.md) — measurement invariance
- [Ordinal Data](guide/dwls.md) — DWLS for Likert-scale data
