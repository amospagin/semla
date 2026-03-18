# Ordinal Data (DWLS)

When your observed variables are ordinal (e.g., Likert scales with 2–7 categories), standard ML estimation assumes continuous data and can produce biased estimates. DWLS uses **polychoric correlations** instead.

## When to Use DWLS

- Likert-scale items (e.g., 1–5 agreement scales)
- Binary items (yes/no)
- Any ordinal categorical data

If your data is continuous (e.g., test scores, reaction times), use the default ML estimator.

## Basic Usage

```python
from semla import cfa

model = """
    f1 =~ item1 + item2 + item3
    f2 =~ item4 + item5 + item6
"""

fit = cfa(model, data=df, estimator="DWLS")
fit.summary()
```

## What Happens Under the Hood

1. **Polychoric correlations** are computed for all pairs of ordinal variables — these estimate the correlation between the latent continuous variables assumed to underlie the ordinal responses
2. The model is fitted via **ML on the polychoric correlation matrix**
3. **Robust standard errors** are computed using a sandwich estimator that accounts for the uncertainty in the polychoric correlations
4. The **chi-square test** is scaled (Satorra-Bentler type) to correct for the non-normality of ordinal data

## Example with Ordinal Data

```python
import pandas as pd
from semla import cfa
from semla.datasets import HolzingerSwineford1939

# Create ordinal version (simulate Likert data)
df = HolzingerSwineford1939()
for col in ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]:
    df[col] = pd.cut(df[col], bins=5, labels=[1, 2, 3, 4, 5]).astype(float)

model = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

fit = cfa(model, data=df, estimator="DWLS")
fit.summary()
```

## Polychoric Correlations

You can also compute polychoric correlations directly:

```python
from semla.polychoric import polychoric_correlation_matrix

R, avar, thresholds = polychoric_correlation_matrix(data.values)
```

!!! note
    Variables with more than 10 unique values are treated as continuous (Pearson correlation is used instead of polychoric). This means DWLS works fine even if your dataset mixes ordinal and continuous variables.
