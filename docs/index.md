# semla

**Structural Equation Modeling with lavaan-style syntax for Python.**

!!! note "Early Development"
    semla is in early development (v0.1.0). The API may change, and results should be validated against established tools like lavaan before use in published research.

semla brings the familiar [lavaan](https://lavaan.ugent.be/) model syntax from R to Python. If you know lavaan, you already know semla.

## Quick Example

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

## Features

- **lavaan syntax** — same `=~`, `~`, `~~` operators you already know
- **ML and DWLS estimation** — maximum likelihood for continuous data, polychoric correlations for ordinal
- **Fit indices** — chi-square, CFI, TLI, RMSEA (with CI), SRMR
- **Standardized solutions** — fully standardized (std.all) and LV-standardized (std.lv)
- **Modification indices** — identify model misspecifications
- **Multi-group analysis** — configural and metric invariance with chi-square difference testing
- **Validated against lavaan** — parameter estimates and standard errors match within 0.005

## Installation

```bash
pip install semla
```

Or from source:

```bash
git clone https://github.com/amospagin/semla.git
cd semla
pip install -e .
```

## Why semla?

There's no mature Python package that lets you specify SEM models with lavaan syntax. If you're a researcher who uses lavaan in R but works in Python for data processing, semla lets you stay in one language.
