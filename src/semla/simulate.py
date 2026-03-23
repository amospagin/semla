"""Monte Carlo power analysis for structural equation models.

Generates data from a population model (multivariate normal from the
RAM-implied covariance) and fits the model repeatedly, collecting
parameter estimates, standard errors, and p-values across replications.

Example
-------
>>> from semla.simulate import simulate_power
>>> model = "f =~ x1 + x2 + x3"
>>> pop = {
...     ("f", "=~", "x2"): 0.8,
...     ("f", "=~", "x3"): 0.8,
...     ("x1", "~~", "x1"): 0.36,
...     ("x2", "~~", "x2"): 0.36,
...     ("x3", "~~", "x3"): 0.36,
...     ("f", "~~", "f"): 1.0,
... }
>>> result = simulate_power(model, pop, n=200, n_replications=500)
>>> result.summary()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .estimation import _model_implied_cov
from .specification import ModelSpecification, build_specification
from .syntax import parse_syntax


@dataclass
class PowerResult:
    """Results from a Monte Carlo power analysis.

    Attributes
    ----------
    n : int
        Sample size per replication.
    n_replications : int
        Total number of replications attempted.
    alpha : float
        Significance threshold used.
    population : dict
        Population parameter values.
    param_info : list[dict]
        Metadata for each free parameter (lhs, op, rhs, pop_value).
    estimates_matrix : np.ndarray
        (n_replications, n_params) matrix of point estimates.
    se_matrix : np.ndarray
        (n_replications, n_params) matrix of standard errors.
    pvalue_matrix : np.ndarray
        (n_replications, n_params) matrix of p-values.
    converged_mask : np.ndarray
        (n_replications,) boolean: True if replication converged.
    """

    n: int
    n_replications: int
    alpha: float
    population: dict
    param_info: list[dict] = field(default_factory=list)
    estimates_matrix: np.ndarray = field(default=None)
    se_matrix: np.ndarray = field(default=None)
    pvalue_matrix: np.ndarray = field(default=None)
    converged_mask: np.ndarray = field(default=None)

    @property
    def convergence_rate(self) -> float:
        """Proportion of replications that converged."""
        return float(np.mean(self.converged_mask))

    def _converged_idx(self) -> np.ndarray:
        return np.where(self.converged_mask)[0]

    def power(self) -> pd.DataFrame:
        """Power for each free parameter: proportion of converged replications
        where p < alpha.

        Returns
        -------
        pd.DataFrame
            Columns: lhs, op, rhs, pop_value, power.
        """
        idx = self._converged_idx()
        if len(idx) == 0:
            powers = np.full(len(self.param_info), np.nan)
        else:
            pvals = self.pvalue_matrix[idx]
            powers = np.mean(pvals < self.alpha, axis=0)

        rows = []
        for i, info in enumerate(self.param_info):
            rows.append({
                "lhs": info["lhs"],
                "op": info["op"],
                "rhs": info["rhs"],
                "pop_value": info["pop_value"],
                "power": powers[i],
            })
        return pd.DataFrame(rows)

    def bias(self) -> pd.DataFrame:
        """Average bias (mean estimate - population value) for each parameter,
        computed over converged replications.

        Returns
        -------
        pd.DataFrame
            Columns: lhs, op, rhs, pop_value, mean_est, bias, rel_bias.
        """
        idx = self._converged_idx()
        if len(idx) == 0:
            mean_est = np.full(len(self.param_info), np.nan)
        else:
            mean_est = np.mean(self.estimates_matrix[idx], axis=0)

        rows = []
        for i, info in enumerate(self.param_info):
            pop = info["pop_value"]
            b = mean_est[i] - pop
            rel_b = b / pop if pop != 0 else np.nan
            rows.append({
                "lhs": info["lhs"],
                "op": info["op"],
                "rhs": info["rhs"],
                "pop_value": pop,
                "mean_est": mean_est[i],
                "bias": b,
                "rel_bias": rel_b,
            })
        return pd.DataFrame(rows)

    def coverage(self) -> pd.DataFrame:
        """95% CI coverage rate for each parameter (proportion of converged
        replications where the population value falls inside the Wald CI).

        Returns
        -------
        pd.DataFrame
            Columns: lhs, op, rhs, pop_value, coverage.
        """
        idx = self._converged_idx()
        if len(idx) == 0:
            covs = np.full(len(self.param_info), np.nan)
        else:
            ests = self.estimates_matrix[idx]
            ses = self.se_matrix[idx]
            lower = ests - 1.96 * ses
            upper = ests + 1.96 * ses
            pop_vals = np.array([info["pop_value"] for info in self.param_info])
            inside = (lower <= pop_vals) & (pop_vals <= upper)
            covs = np.mean(inside, axis=0)

        rows = []
        for i, info in enumerate(self.param_info):
            rows.append({
                "lhs": info["lhs"],
                "op": info["op"],
                "rhs": info["rhs"],
                "pop_value": info["pop_value"],
                "coverage": covs[i],
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = []
        lines.append("Monte Carlo Power Analysis")
        lines.append(f"  Sample size (n):     {self.n}")
        lines.append(f"  Replications:        {self.n_replications}")
        lines.append(f"  Converged:           {int(np.sum(self.converged_mask))} "
                      f"({self.convergence_rate:.1%})")
        lines.append(f"  Alpha:               {self.alpha}")
        lines.append("")

        pw = self.power()
        bi = self.bias()
        cv = self.coverage()

        lines.append(f"{'Parameter':<25s} {'Pop':>8s} {'Power':>8s} "
                      f"{'Bias':>8s} {'Coverage':>8s}")
        lines.append("-" * 65)
        for i, info in enumerate(self.param_info):
            label = f"{info['lhs']} {info['op']} {info['rhs']}"
            pop_str = f"{info['pop_value']:.3f}"
            pw_str = f"{pw.iloc[i]['power']:.3f}" if not np.isnan(pw.iloc[i]['power']) else "   NA"
            bi_str = f"{bi.iloc[i]['bias']:.4f}" if not np.isnan(bi.iloc[i]['bias']) else "   NA"
            cv_str = f"{cv.iloc[i]['coverage']:.3f}" if not np.isnan(cv.iloc[i]['coverage']) else "   NA"
            lines.append(f"{label:<25s} {pop_str:>8s} {pw_str:>8s} "
                          f"{bi_str:>8s} {cv_str:>8s}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _build_population_matrices(
    spec: ModelSpecification,
    population: dict[tuple[str, str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Set population values into A and S matrices.

    Population dict keys are (lhs, op, rhs) tuples. Fixed parameters
    (e.g. first loading = 1.0) are already set in spec.A_values / S_values.
    Free parameters are overwritten from the population dict.

    Returns
    -------
    A, S : np.ndarray
        Population RAM matrices.
    """
    A = spec.A_values.copy()
    S = spec.S_values.copy()

    for (lhs, op, rhs), value in population.items():
        if op == "=~":
            row = spec._idx(rhs)
            col = spec._idx(lhs)
            A[row, col] = value
        elif op == "~":
            row = spec._idx(lhs)
            col = spec._idx(rhs)
            A[row, col] = value
        elif op == "~~":
            i = spec._idx(lhs)
            j = spec._idx(rhs)
            S[i, j] = value
            S[j, i] = value
        else:
            raise ValueError(f"Unsupported operator in population dict: {op!r}")

    return A, S


def _generate_data(
    A: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate multivariate normal data from the population model.

    Parameters
    ----------
    A, S, F : np.ndarray
        Population RAM matrices.
    n : int
        Number of observations.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        (n, p) data matrix for observed variables.
    """
    sigma = _model_implied_cov(A, S, F)
    if sigma is None:
        raise ValueError("Cannot compute implied covariance: (I - A) is singular.")

    # Ensure positive definite (symmetrize and check)
    sigma = (sigma + sigma.T) / 2.0
    eigvals = np.linalg.eigvalsh(sigma)
    if np.any(eigvals <= 0):
        raise ValueError(
            f"Population implied covariance is not positive definite. "
            f"Smallest eigenvalue: {eigvals.min():.6e}. "
            f"Check population parameter values."
        )

    return rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=n)


def simulate_power(
    model: str,
    population: dict[tuple[str, str, str], float],
    n: int = 200,
    n_replications: int = 500,
    estimator: str = "ML",
    seed: int = 42,
    alpha: float = 0.05,
) -> PowerResult:
    """Monte Carlo power analysis for a structural equation model.

    Generates data from a known population model and fits the model to
    each generated dataset, collecting parameter estimates, SEs, and
    p-values across replications.

    Parameters
    ----------
    model : str
        Model syntax in lavaan format.
    population : dict
        True parameter values. Keys are ``(lhs, op, rhs)`` tuples, e.g.
        ``("f", "=~", "x2"): 0.8``. The first loading per factor is
        automatically fixed to 1.0 and should NOT appear here.
    n : int
        Sample size per replication (default 200).
    n_replications : int
        Number of Monte Carlo replications (default 500).
    estimator : str
        Estimation method (currently only ``"ML"`` is supported).
    seed : int
        Random seed for reproducibility (default 42).
    alpha : float
        Significance level for power calculation (default 0.05).

    Returns
    -------
    PowerResult
        Object with ``power()``, ``bias()``, ``coverage()``, and
        ``summary()`` methods.
    """
    from .model import cfa  # local import to avoid circular dependency

    if estimator != "ML":
        raise NotImplementedError(f"Only ML estimator is supported, got {estimator!r}")

    rng = np.random.default_rng(seed)

    # --- Build specification to get RAM matrices ---
    tokens = parse_syntax(model)
    # We need observed variable names. Since we don't have data yet,
    # we infer them: any variable on the RHS of =~ or any variable
    # referenced that is NOT on the LHS of =~.
    latent_vars = {tok.lhs for tok in tokens if tok.op == "=~"}
    obs_vars = []
    for tok in tokens:
        for term in tok.rhs:
            if term.var not in latent_vars and term.var not in obs_vars:
                obs_vars.append(term.var)
        if tok.op in ("~", "~~") and tok.lhs not in latent_vars:
            if tok.lhs not in obs_vars:
                obs_vars.append(tok.lhs)

    spec = build_specification(tokens, obs_vars)

    # --- Build population matrices ---
    A_pop, S_pop = _build_population_matrices(spec, population)

    # --- Identify free parameters and their population values ---
    param_info = []
    for p in spec.params:
        if not p.free:
            continue
        # Determine population value from the matrices
        if p.op == "=~":
            row = spec._idx(p.rhs)
            col = spec._idx(p.lhs)
            pop_val = A_pop[row, col]
        elif p.op == "~":
            row = spec._idx(p.lhs)
            col = spec._idx(p.rhs)
            pop_val = A_pop[row, col]
        elif p.op == "~~":
            i = spec._idx(p.lhs)
            j = spec._idx(p.rhs)
            pop_val = S_pop[i, j]
        else:
            pop_val = 0.0

        param_info.append({
            "lhs": p.lhs,
            "op": p.op,
            "rhs": p.rhs,
            "pop_value": pop_val,
        })

    n_params = len(param_info)
    estimates_matrix = np.full((n_replications, n_params), np.nan)
    se_matrix = np.full((n_replications, n_params), np.nan)
    pvalue_matrix = np.full((n_replications, n_params), np.nan)
    converged_mask = np.zeros(n_replications, dtype=bool)

    # --- Monte Carlo loop ---
    for rep in range(n_replications):
        # Generate data
        data_array = _generate_data(A_pop, S_pop, spec.F, n, rng)
        df = pd.DataFrame(data_array, columns=spec.observed_vars)

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = cfa(model, data=df)
            except Exception:
                converged_mask[rep] = False
                continue

        converged_mask[rep] = fit.converged
        if not fit.converged:
            continue

        # Extract estimates
        est_df = fit.estimates()
        free_rows = est_df[est_df["free"]].reset_index(drop=True)

        for i, info in enumerate(param_info):
            # Match parameter by lhs, op, rhs
            mask = (
                (free_rows["lhs"] == info["lhs"])
                & (free_rows["op"] == info["op"])
                & (free_rows["rhs"] == info["rhs"])
            )
            matched = free_rows[mask]
            if len(matched) == 1:
                row = matched.iloc[0]
                estimates_matrix[rep, i] = row["est"]
                se_matrix[rep, i] = row["se"]
                pvalue_matrix[rep, i] = row["pvalue"]

    return PowerResult(
        n=n,
        n_replications=n_replications,
        alpha=alpha,
        population=population,
        param_info=param_info,
        estimates_matrix=estimates_matrix,
        se_matrix=se_matrix,
        pvalue_matrix=pvalue_matrix,
        converged_mask=converged_mask,
    )
