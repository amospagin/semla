"""User-facing Model class and convenience functions."""

from __future__ import annotations

import pandas as pd

from .estimation import estimate
from .results import ModelResults
from .specification import build_specification
from .syntax import parse_syntax


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

        # Build specification
        auto_cov_latent = kwargs.get("auto_cov_latent", True)
        self.spec = build_specification(
            self.tokens,
            data.columns.tolist(),
            auto_cov_latent=auto_cov_latent,
        )

        # Estimate
        est_result = estimate(self.spec, data)

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

    @property
    def converged(self) -> bool:
        """Whether the optimizer converged."""
        return self.results.converged


def cfa(model: str, data: pd.DataFrame, **kwargs) -> Model:
    """Fit a Confirmatory Factor Analysis model.

    Convenience function matching lavaan::cfa(). Automatically adds
    covariances between latent variables.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    data : pd.DataFrame
        Data with columns matching observed variables.

    Returns
    -------
    Model
        Fitted model object.
    """
    kwargs.setdefault("auto_cov_latent", True)
    return Model(model, data, **kwargs)


def sem(model: str, data: pd.DataFrame, **kwargs) -> Model:
    """Fit a Structural Equation Model.

    Convenience function matching lavaan::sem(). Does NOT auto-add
    covariances between latent variables (unlike cfa()).

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    data : pd.DataFrame
        Data with columns matching observed variables.

    Returns
    -------
    Model
        Fitted model object.
    """
    kwargs.setdefault("auto_cov_latent", False)
    return Model(model, data, **kwargs)
