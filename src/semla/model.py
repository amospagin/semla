"""User-facing Model class and convenience functions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .estimation import estimate
from .results import ModelResults
from .specification import build_specification
from .syntax import parse_syntax


def _validate_data(tokens, data: pd.DataFrame) -> None:
    """Validate that data is compatible with the model specification."""
    data_cols = set(data.columns)

    # Collect all observed variables referenced in the model
    latent_vars = {tok.lhs for tok in tokens if tok.op == "=~"}
    referenced_vars = set()
    for tok in tokens:
        for term in tok.rhs:
            if term.var not in latent_vars:
                referenced_vars.add(term.var)
        if tok.op in ("~", "~~") and tok.lhs not in latent_vars:
            referenced_vars.add(tok.lhs)

    # Check for missing variables
    missing = referenced_vars - data_cols
    if missing:
        missing_sorted = sorted(missing)
        available = sorted(data_cols)
        raise ValueError(
            f"Variable(s) not found in data: {missing_sorted}. "
            f"Available columns: {available}"
        )

    # Check for constant columns (zero variance)
    for var in referenced_vars:
        if var in data_cols and data[var].std() == 0:
            raise ValueError(
                f"Variable '{var}' has zero variance (constant column). "
                f"Remove it from the model or check your data."
            )

    # Check minimum sample size
    n_obs = len(data)
    if n_obs < 10:
        warnings.warn(
            f"Very small sample size (n={n_obs}). Results may be unreliable.",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_syntax(tokens) -> None:
    """Validate model syntax for common specification errors."""
    # Check for duplicate indicators within a single factor
    for tok in tokens:
        if tok.op == "=~":
            seen = set()
            for term in tok.rhs:
                if term.var in seen:
                    raise ValueError(
                        f"Duplicate indicator '{term.var}' in definition of "
                        f"'{tok.lhs}'. Each indicator should appear only once "
                        f"per factor."
                    )
                seen.add(term.var)

    # Check for single-indicator factors
    latent_indicators: dict[str, list[str]] = {}
    for tok in tokens:
        if tok.op == "=~":
            lv = tok.lhs
            if lv not in latent_indicators:
                latent_indicators[lv] = []
            latent_indicators[lv].extend(t.var for t in tok.rhs)

    for lv, indicators in latent_indicators.items():
        if len(indicators) < 2:
            warnings.warn(
                f"Latent variable '{lv}' has only {len(indicators)} "
                f"indicator(s). At least 2 are needed for identification "
                f"(3+ recommended).",
                RuntimeWarning,
                stacklevel=3,
            )


class Model:
    """A Structural Equation Model.

    Parameters
    ----------
    syntax : str
        Model specification in lavaan syntax.
    data : pd.DataFrame
        Data with columns matching observed variables.

    Examples
    --------
    >>> import pandas as pd
    >>> model = Model('''
    ...     visual  =~ x1 + x2 + x3
    ...     textual =~ x4 + x5 + x6
    ...     speed   =~ x7 + x8 + x9
    ... ''', data=df)
    >>> model.summary()
    """

    def __init__(self, syntax: str, data: pd.DataFrame, **kwargs):
        self.syntax_str = syntax
        self.data = data

        # Parse
        self.tokens = parse_syntax(syntax)

        # Validate
        _validate_syntax(self.tokens)
        _validate_data(self.tokens, data)

        # Build specification
        auto_cov_latent = kwargs.get("auto_cov_latent", True)
        self.spec = build_specification(
            self.tokens,
            data.columns.tolist(),
            auto_cov_latent=auto_cov_latent,
        )

        # Estimate
        estimator = kwargs.get("estimator", "ML").upper()
        if estimator == "ML":
            est_result = estimate(self.spec, data)
        elif estimator == "DWLS":
            from .dwls import estimate_dwls
            est_result = estimate_dwls(self.spec, data)
        else:
            raise ValueError(
                f"Unknown estimator '{estimator}'. Use 'ML' or 'DWLS'."
            )

        # Build results
        self.results = ModelResults(est_result)

    def summary(self) -> str:
        """Print and return a lavaan-style summary."""
        return self.results.summary()

    def fit_indices(self) -> dict:
        """Return fit indices as a dictionary."""
        return self.results.fit_indices()

    def estimates(self) -> pd.DataFrame:
        """Return parameter estimates as a DataFrame."""
        return self.results.estimates()

    def modindices(self, min_mi: float = 0.0, sort: bool = True) -> pd.DataFrame:
        """Return modification indices for fixed parameters.

        Parameters
        ----------
        min_mi : float
            Only return parameters with MI >= this value.
        sort : bool
            Sort by MI descending.
        """
        return self.results.modindices(min_mi=min_mi, sort=sort)

    def r_squared(self) -> dict:
        """Return R-squared for endogenous variables."""
        return self.results.r_squared()

    def standardized_estimates(self, type: str = "std.all") -> pd.DataFrame:
        """Return standardized parameter estimates.

        Parameters
        ----------
        type : str
            ``"std.all"`` (fully standardized) or ``"std.lv"`` (by LV SD only).
        """
        return self.results.standardized_estimates(type=type)

    @property
    def converged(self) -> bool:
        """Whether the optimizer converged."""
        return self.results.converged


class MultiGroupModel:
    """A multi-group Structural Equation Model.

    Parameters
    ----------
    syntax : str
        Model specification in lavaan syntax.
    data : pd.DataFrame
        Data with a grouping column.
    group : str
        Column name defining groups.
    invariance : str
        ``"configural"`` or ``"metric"``.
    """

    def __init__(self, syntax: str, data: pd.DataFrame, group: str,
                 invariance: str = "configural", **kwargs):
        from .multigroup import build_multigroup_spec, estimate_multigroup
        from .multigroup_results import MultiGroupResults

        self.syntax_str = syntax
        self.data = data
        self.group_col = group
        self.invariance = invariance

        tokens = parse_syntax(syntax)
        _validate_syntax(tokens)

        auto_cov_latent = kwargs.get("auto_cov_latent", True)
        self.mg_spec = build_multigroup_spec(
            tokens, data, group, invariance=invariance,
            auto_cov_latent=auto_cov_latent,
        )

        est_result = estimate_multigroup(self.mg_spec)
        self.results = MultiGroupResults(est_result)

    def summary(self) -> str:
        return self.results.summary()

    def fit_indices(self) -> dict:
        return self.results.fit_indices()

    def estimates(self) -> pd.DataFrame:
        return self.results.estimates()

    @property
    def converged(self) -> bool:
        return self.results.converged


def cfa(model: str, data: pd.DataFrame, group: str = None, **kwargs):
    """Fit a Confirmatory Factor Analysis model.

    Convenience function matching lavaan::cfa(). Automatically adds
    covariances between latent variables.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    data : pd.DataFrame
        Data with columns matching observed variables.
    group : str, optional
        Column name for multi-group analysis.

    Returns
    -------
    Model or MultiGroupModel
        Fitted model object.
    """
    kwargs.setdefault("auto_cov_latent", True)
    if group is not None:
        return MultiGroupModel(model, data, group=group, **kwargs)
    return Model(model, data, **kwargs)


def sem(model: str, data: pd.DataFrame, group: str = None, **kwargs):
    """Fit a Structural Equation Model.

    Convenience function matching lavaan::sem(). Does NOT auto-add
    covariances between latent variables (unlike cfa()).

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    data : pd.DataFrame
        Data with columns matching observed variables.
    group : str, optional
        Column name for multi-group analysis.

    Returns
    -------
    Model or MultiGroupModel
        Fitted model object.
    """
    kwargs.setdefault("auto_cov_latent", False)
    if group is not None:
        return MultiGroupModel(model, data, group=group, **kwargs)
    return Model(model, data, **kwargs)
