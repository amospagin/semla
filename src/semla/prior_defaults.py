"""Data-adaptive and weak prior defaults for Bayesian SEM.

Resolves user-supplied prior specifications (string presets, matrix-level
overrides, per-parameter overrides) into a concrete prior for every free
parameter in a :class:`~semla.specification.ModelSpecification`.

Priority (lowest to highest):
    1. Adaptive / weak defaults (base tier)
    2. Matrix-level overrides  (e.g. ``{"loadings": Normal(0, 1)}``)
    3. Per-parameter overrides (e.g. ``{"f1=~x2": Normal(0.7, 0.2)}``)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .priors import (
    Prior,
    Normal,
    StudentT,
    HalfCauchy,
    InverseGamma,
    LKJ,
    Gamma,
)
from .specification import ModelSpecification, ParamInfo

# ── type aliases ────────────────────────────────────────────────────────
PriorSpec = Union[str, dict[str, Prior], None]

# ── matrix-level keys ───────────────────────────────────────────────────
MATRIX_KEYS = {
    "loadings",
    "regressions",
    "residual_variances",
    "factor_variances",
    "covariances",
    "intercepts",
}


# ── helpers ─────────────────────────────────────────────────────────────

def _param_key(p: ParamInfo) -> str:
    """Return the canonical string key for a parameter (e.g. ``'f1=~x2'``)."""
    return f"{p.lhs}{p.op}{p.rhs}"


def _matrix_category(p: ParamInfo, spec: ModelSpecification) -> str:
    """Classify a free parameter into one of the matrix-level categories."""
    if p.op == "=~":
        return "loadings"
    if p.op == "~":
        return "regressions"
    if p.op == "~1":
        return "intercepts"
    # op == "~~"
    if p.lhs == p.rhs:
        if p.lhs in spec.latent_vars:
            return "factor_variances"
        return "residual_variances"
    return "covariances"


# ── adaptive defaults (brms-style, scaled by observed SDs) ─────────────

def _observed_sds(spec: ModelSpecification, data: np.ndarray) -> dict[str, float]:
    """Map each observed variable to its sample SD."""
    sds: dict[str, float] = {}
    for i, var in enumerate(spec.observed_vars):
        col = data[:, i]
        col_clean = col[~np.isnan(col)]
        sds[var] = float(np.std(col_clean, ddof=1)) if len(col_clean) > 1 else 1.0
    return sds


def _adaptive_prior(
    p: ParamInfo,
    category: str,
    obs_sds: dict[str, float],
    spec: ModelSpecification,
) -> Prior:
    """Return a data-adaptive prior for a single free parameter.

    Scaling follows the brms convention: use observed SDs to set prior
    widths so that priors are weakly informative relative to the data
    scale.
    """
    # median SD across all observed indicators as a scale reference
    median_sd = float(np.median(list(obs_sds.values()))) if obs_sds else 1.0

    if category == "loadings":
        # Normal centred at 0, width ~ 2.5 * indicator SD
        rhs_sd = obs_sds.get(p.rhs, median_sd)
        return Normal(mu=0.0, sigma=2.5 * rhs_sd)

    if category == "regressions":
        return Normal(mu=0.0, sigma=2.5 * median_sd)

    if category == "residual_variances":
        var_sd = obs_sds.get(p.lhs, median_sd)
        return InverseGamma(concentration=2.0, rate=var_sd ** 2)

    if category == "factor_variances":
        return InverseGamma(concentration=2.0, rate=median_sd ** 2)

    if category == "covariances":
        return Normal(mu=0.0, sigma=2.5 * median_sd ** 2)

    if category == "intercepts":
        var_sd = obs_sds.get(p.lhs, median_sd)
        return Normal(mu=0.0, sigma=10.0 * var_sd)

    # fallback
    return Normal(mu=0.0, sigma=10.0)  # pragma: no cover


# ── weak informative preset ────────────────────────────────────────────

_WEAK_DEFAULTS: dict[str, Prior] = {
    "loadings": Normal(mu=0.0, sigma=10.0),
    "regressions": Normal(mu=0.0, sigma=10.0),
    "residual_variances": InverseGamma(concentration=0.01, rate=0.01),
    "factor_variances": InverseGamma(concentration=0.01, rate=0.01),
    "covariances": Normal(mu=0.0, sigma=100.0),
    "intercepts": Normal(mu=0.0, sigma=100.0),
}


def _weak_prior(category: str) -> Prior:
    """Return a fixed wide (weak informative) prior."""
    return _WEAK_DEFAULTS[category]


# ── public API ──────────────────────────────────────────────────────────

def resolve_priors(
    spec: ModelSpecification,
    data: np.ndarray,
    priors: PriorSpec = None,
) -> dict[str, Prior]:
    """Resolve a user-supplied prior specification to a per-parameter dict.

    Parameters
    ----------
    spec : ModelSpecification
        The built RAM specification (must contain ``params``).
    data : np.ndarray
        Observed data matrix (n_obs × p_observed), columns matching
        ``spec.observed_vars``.
    priors : str, dict, or None
        * ``None`` (default) — data-adaptive priors scaled by observed SDs.
        * ``"weak"`` — fixed wide priors (InverseGamma(0.01, 0.01) for
          variances, Normal(0, 10) for loadings, etc.).
        * ``dict`` — mix of matrix-level and per-parameter overrides.
          Matrix-level keys: ``"loadings"``, ``"regressions"``,
          ``"residual_variances"``, ``"factor_variances"``,
          ``"covariances"``, ``"intercepts"``.
          Per-parameter keys use lavaan syntax: ``"f1=~x2"``.

    Returns
    -------
    dict[str, Prior]
        Mapping from parameter key (e.g. ``"f1=~x2"``) to a
        :class:`~semla.priors.Prior` instance for every free parameter.
    """
    # separate overrides into tiers
    matrix_overrides: dict[str, Prior] = {}
    param_overrides: dict[str, Prior] = {}
    use_weak = False

    if isinstance(priors, str):
        if priors.lower() == "weak":
            use_weak = True
        else:
            raise ValueError(
                f"Unknown prior preset: {priors!r}. Use 'weak' or a dict."
            )
    elif isinstance(priors, dict):
        for key, val in priors.items():
            if not isinstance(val, Prior):
                raise TypeError(
                    f"Prior for {key!r} must be a Prior instance, got {type(val).__name__}"
                )
            if key in MATRIX_KEYS:
                matrix_overrides[key] = val
            else:
                param_overrides[key] = val

    # pre-compute observed SDs for adaptive scaling
    obs_sds = _observed_sds(spec, data) if not use_weak else {}

    # resolve for every free parameter
    result: dict[str, Prior] = {}
    for p in spec.params:
        if not p.free:
            continue
        key = _param_key(p)
        category = _matrix_category(p, spec)

        # tier 1: base default
        if use_weak:
            prior = _weak_prior(category)
        else:
            prior = _adaptive_prior(p, category, obs_sds, spec)

        # tier 2: matrix-level override
        if category in matrix_overrides:
            prior = matrix_overrides[category]

        # tier 3: per-parameter override
        if key in param_overrides:
            prior = param_overrides[key]

        result[key] = prior

    # warn about unused per-parameter overrides
    used_keys = set(result.keys())
    unused = set(param_overrides.keys()) - used_keys
    if unused:
        import warnings
        warnings.warn(
            f"Prior overrides for unknown parameters ignored: {unused}",
            UserWarning,
            stacklevel=2,
        )

    return result
