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
        if tok.op == ":=":
            continue  # defined params reference labels, not data variables
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
        meanstructure = kwargs.get("meanstructure", False)
        # Auto-detect from syntax
        if any(tok.op == "~1" for tok in self.tokens):
            meanstructure = True

        # Filter out := tokens (defined params, not model specification)
        model_tokens = [tok for tok in self.tokens if tok.op != ":="]

        int_ov_free = kwargs.get("int_ov_free", True)
        int_lv_free = kwargs.get("int_lv_free", False)

        self.spec = build_specification(
            model_tokens,
            data.columns.tolist(),
            auto_cov_latent=auto_cov_latent,
            meanstructure=meanstructure,
            int_ov_free=int_ov_free,
            int_lv_free=int_lv_free,
        )

        # Set starting values for intercepts from sample means
        if self.spec.meanstructure and self.spec.m_values is not None:
            for i, var in enumerate(self.spec.observed_vars):
                idx = self.spec._idx(var)
                if self.spec.m_free[idx]:
                    self.spec.m_values[idx] = data[var].mean()

            # For growth models (int_lv_free=True): set latent mean starting
            # values from observed means.  Intercept factor ≈ mean(y1),
            # slope factor ≈ average per-timepoint change.
            if int_lv_free:
                obs = self.spec.observed_vars
                obs_means = [data[v].mean() for v in obs]
                for lv in self.spec.latent_vars:
                    idx = self.spec._idx(lv)
                    if self.spec.m_free[idx]:
                        # Use first observed mean as intercept start,
                        # average change as slope start
                        self.spec.m_values[idx] = obs_means[0]
                if len(self.spec.latent_vars) >= 2:
                    # Second latent var (slope): average change per unit
                    n_t = len(obs)
                    if n_t > 1:
                        slope_start = (obs_means[-1] - obs_means[0]) / (n_t - 1)
                        slope_lv = self.spec.latent_vars[1]
                        slope_idx = self.spec._idx(slope_lv)
                        if self.spec.m_free[slope_idx]:
                            self.spec.m_values[slope_idx] = slope_start

        # Estimate
        # Handle missing data
        missing = kwargs.get("missing", "listwise")
        obs_vars = self.spec.observed_vars
        has_missing = data[obs_vars].isna().any().any()

        if has_missing and missing == "listwise":
            n_total = len(data)
            data = data.dropna(subset=obs_vars)
            n_complete = len(data)
            if n_complete < n_total:
                warnings.warn(
                    f"Data contains missing values. {n_total - n_complete} of {n_total} "
                    f"cases dropped (listwise deletion). "
                    f"Consider using missing='fiml' to use all available information.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.data = data

        # Estimate
        estimator = kwargs.get("estimator", "ML").upper()

        if estimator == "BAYES":
            self._fit_bayes(data, kwargs)
            return

        if missing == "fiml":
            from .fiml import estimate_fiml
            # FIML requires mean structure
            if not self.spec.meanstructure:
                model_tokens = [tok for tok in self.tokens if tok.op != ":="]
                self.spec = build_specification(
                    model_tokens, data.columns.tolist(),
                    auto_cov_latent=kwargs.get("auto_cov_latent", True),
                    meanstructure=True,
                )
                for i, var in enumerate(self.spec.observed_vars):
                    idx = self.spec._idx(var)
                    if self.spec.m_free[idx]:
                        self.spec.m_values[idx] = data[var].mean()
            est_result = estimate_fiml(self.spec, data)
        elif estimator in ("ML", "MLR"):
            est_result = estimate(self.spec, data)
            if estimator == "MLR":
                est_result._estimator_type = "MLR"
        elif estimator == "DWLS":
            from .dwls import estimate_dwls
            est_result = estimate_dwls(self.spec, data)
        else:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                "Use 'ML', 'MLR', 'DWLS', or 'bayes'."
            )

        # Build results
        # Extract defined parameters (:=) from tokens
        from .defined import extract_defined_params
        self._defined_params = extract_defined_params(self.tokens)
        self.results = ModelResults(est_result, defined_params=self._defined_params)

    def _fit_bayes(self, data: pd.DataFrame, kwargs: dict):
        """Run Bayesian estimation via NumPyro MCMC."""
        try:
            from .bayes import run_mcmc
        except ImportError:
            raise ImportError(
                "numpyro is required for estimator='bayes'. "
                "Install it with:  pip install semla[bayes]"
            ) from None

        obs_vars = self.spec.observed_vars
        data_array = data[obs_vars].values.astype(float)

        bayes_kwargs = {
            "priors": kwargs.get("priors", None),
            "num_warmup": kwargs.get("warmup", kwargs.get("num_warmup", 1000)),
            "num_samples": kwargs.get("draws", kwargs.get("num_samples", 1000)),
            "num_chains": kwargs.get("chains", kwargs.get("num_chains", 4)),
            "cores": kwargs.get("cores", None),
            "seed": kwargs.get("seed", 0),
            "positive_loadings": kwargs.get("positive_loadings", True),
            "target_accept_prob": kwargs.get("adapt_delta",
                                             kwargs.get("target_accept_prob", 0.8)),
            "adapt_convergence": kwargs.get("adapt_convergence", True),
            "progress_bar": kwargs.get("progress_bar", True),
        }

        self.results = run_mcmc(self.spec, data_array, **bayes_kwargs)

    def summary(self) -> str:
        """Print and return a lavaan-style summary."""
        return self.results.summary()

    def fit_indices(self) -> dict:
        """Return fit indices as a dictionary."""
        return self.results.fit_indices()

    def estimates(self) -> pd.DataFrame:
        """Return parameter estimates as a DataFrame."""
        return self.results.estimates()

    def fitted(self) -> dict:
        """Return model-implied covariance matrix and mean vector.

        Returns
        -------
        dict
            ``"cov"`` : pd.DataFrame, ``"mean"`` : pd.Series or None.
        """
        return self.results.fitted()

    def vcov(self) -> pd.DataFrame:
        """Return the parameter variance-covariance matrix.

        Returns
        -------
        pd.DataFrame
            Square matrix with rows/columns labeled by parameter.
        """
        return self.results.vcov()

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

    def defined_estimates(self) -> pd.DataFrame:
        """Return estimates for user-defined parameters (:= operator)."""
        return self.results.defined_estimates()

    def residuals(self, type: str = "raw") -> np.ndarray:
        """Return residual covariance matrix (observed - implied)."""
        return self.results.residuals(type=type)

    def r_squared(self) -> dict:
        """Return R-squared for endogenous variables."""
        return self.results.r_squared()

    def reliability(self) -> dict:
        """Return reliability (omega, alpha) for each factor."""
        return self.results.reliability()

    def bootstrap(self, nboot: int = 1000, seed: int = None) -> pd.DataFrame:
        """Bootstrap confidence intervals for all parameters.

        Resamples data with replacement, refits the model, and returns
        the distribution of parameter estimates.

        Parameters
        ----------
        nboot : int
            Number of bootstrap replications.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: lhs, op, rhs, est, se_boot, ci_lower, ci_upper, pvalue_boot.
        """
        import warnings as _warnings
        rng = np.random.default_rng(seed)
        n = len(self.data)

        # Original estimates as reference
        orig_est = self.estimates()
        orig_est = orig_est[orig_est["free"]].copy()
        n_params = len(orig_est)

        # Collect bootstrap estimates
        boot_matrix = np.full((nboot, n_params), np.nan)
        model_tokens = [tok for tok in self.tokens if tok.op != ":="]

        for b in range(nboot):
            idx = rng.integers(0, n, size=n)
            boot_data = self.data.iloc[idx].reset_index(drop=True)

            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    boot_spec = build_specification(
                        model_tokens,
                        boot_data.columns.tolist(),
                        auto_cov_latent=self.spec.meanstructure or True,
                        meanstructure=self.spec.meanstructure,
                    )
                    # Copy constraint map if present
                    boot_spec._constraint_map = self.spec._constraint_map

                    # Set starting values from original solution
                    boot_spec.A_values = self.spec.A_values.copy()
                    boot_spec.S_values = self.spec.S_values.copy()
                    if self.spec.meanstructure:
                        boot_spec.m_values = self.spec.m_values.copy()

                    boot_result = estimate(boot_spec, boot_data)
                    if boot_result.converged:
                        # Extract free parameter values in the same order
                        boot_est_df = ModelResults(boot_result).estimates()
                        boot_free = boot_est_df[boot_est_df["free"]]
                        if len(boot_free) == n_params:
                            boot_matrix[b, :] = boot_free["est"].values
            except Exception:
                continue

        # Compute bootstrap statistics
        valid = ~np.all(np.isnan(boot_matrix), axis=1)
        n_valid = valid.sum()

        rows = []
        for i, (_, row) in enumerate(orig_est.iterrows()):
            boot_vals = boot_matrix[valid, i] if n_valid > 0 else np.array([])
            boot_vals = boot_vals[~np.isnan(boot_vals)]

            if len(boot_vals) > 10:
                se_boot = np.std(boot_vals, ddof=1)
                ci_lower = np.percentile(boot_vals, 2.5)
                ci_upper = np.percentile(boot_vals, 97.5)
                # Bootstrap p-value (proportion of samples crossing zero)
                if row["est"] > 0:
                    p_boot = 2 * np.mean(boot_vals <= 0)
                else:
                    p_boot = 2 * np.mean(boot_vals >= 0)
                p_boot = min(p_boot, 1.0)
            else:
                se_boot = ci_lower = ci_upper = p_boot = np.nan

            rows.append({
                "lhs": row["lhs"], "op": row["op"], "rhs": row["rhs"],
                "est": row["est"],
                "se_boot": se_boot,
                "ci.lower": ci_lower,
                "ci.upper": ci_upper,
                "pvalue_boot": p_boot,
                "n_valid": len(boot_vals),
            })

        return pd.DataFrame(rows)

    def predict(self, data: pd.DataFrame = None, method: str = "regression") -> pd.DataFrame:
        """Predict factor scores.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to predict on. Defaults to the training data.
        method : str
            ``"regression"`` or ``"bartlett"``.
        """
        if data is None:
            data = self.data
        return self.results.factor_scores(data, method=method)

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
        # Scalar and strict invariance require mean structure
        meanstructure = kwargs.get("meanstructure", False)
        if invariance in ("scalar", "strict"):
            meanstructure = True

        # Filter := tokens
        model_tokens = [tok for tok in tokens if tok.op != ":="]

        self.mg_spec = build_multigroup_spec(
            model_tokens, data, group, invariance=invariance,
            auto_cov_latent=auto_cov_latent,
            meanstructure=meanstructure,
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


def growth(model: str, data: pd.DataFrame, **kwargs):
    """Fit a latent growth curve model.

    Convenience function matching lavaan::growth(). Sets
    ``meanstructure=True`` with observed intercepts fixed to 0 and
    latent means (intercept, slope) freely estimated.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format with fixed loadings, e.g.::

            i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
            s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4

    data : pd.DataFrame
        Data with columns matching observed variables.

    Returns
    -------
    Model
        Fitted growth model object.
    """
    kwargs["meanstructure"] = True
    kwargs["int_ov_free"] = False
    kwargs["int_lv_free"] = True
    kwargs.setdefault("auto_cov_latent", True)
    return Model(model, data, **kwargs)
