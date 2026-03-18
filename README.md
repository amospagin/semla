# semla

**Structural Equation Modeling with lavaan-style syntax for Python.**

semla brings the familiar lavaan model syntax from R to Python, making it easy to specify and estimate CFA and SEM models.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pandas as pd
from semla import cfa

# Define model using lavaan syntax
model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

# Fit the model
fit = cfa(model, data=df)

# View results
fit.summary()

# Get fit indices
fit.fit_indices()

# Get parameter estimates as DataFrame
fit.estimates()
```

## Model Syntax

semla uses the same operators as lavaan:

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Latent variable definition | `f1 =~ x1 + x2 + x3` |
| `~` | Regression | `y ~ x1 + x2` |
| `~~` | (Co)variance | `x1 ~~ x2` |
| `~1` | Intercept | `y ~1` |

### Modifiers

- Fixed values: `f1 =~ 1*x1 + x2 + x3` (fix loading to 1.0)
- Labels: `f1 =~ a*x1 + b*x2`

## Functions

- **`cfa(model, data)`** — Confirmatory Factor Analysis (auto-adds latent covariances)
- **`sem(model, data)`** — Structural Equation Model
- **`Model(model, data)`** — Direct model fitting

## Fit Indices

- Chi-square test statistic and p-value
- CFI (Comparative Fit Index)
- TLI (Tucker-Lewis Index)
- RMSEA with 90% confidence interval
- SRMR (Standardized Root Mean Square Residual)

## Dependencies

- numpy
- scipy
- pandas

## License

MIT
