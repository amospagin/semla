"""Fit indices, parameter tables, and lavaan-style summary output."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import EstimationResult, _compute_se, _model_implied_cov, ml_objective
from .specification import ModelSpecification, build_specification
from .syntax import FormulaToken, parse_syntax


class ModelResults:
    """Container for estimated model results and fit statistics."""

    def __init__(self, est_result: EstimationResult):
        self._est = est_result
        self._spec = est_result.spec
        self._theta = est_result.theta
        self._sample_cov = est_result.sample_cov
        self._n_obs = est_result.n_obs

        # Compute standard errors
        self._se = _compute_se(
            self._theta, self._spec, self._sample_cov, self._n_obs
        )

        # Compute fit indices
        self._compute_fit_indices()

    @property
    def converged(self) -> bool:
        return self._est.converged

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_free(self) -> int:
        return self._spec.n_free

    # --- Fit indices ---

    def _compute_fit_indices(self):
        p = self._spec.n_obs
        n = self._n_obs

        # Chi-square
        self.fmin = self._est.fmin
        self.chi_square = (n - 1) * self.fmin
        self.df = p * (p + 1) // 2 - self._spec.n_free
        self.p_value = 1.0 - stats.chi2.cdf(self.chi_square, self.df) if self.df > 0 else np.nan

        # Null (independence) model for baseline comparisons
        null_chi, null_df = self._fit_null_model()
        self._null_chi = null_chi
        self._null_df = null_df

        # CFI
        num = max(self.chi_square - self.df, 0)
        den = max(null_chi - null_df, 0)
        self.cfi = 1.0 - num / den if den > 0 else 1.0

        # TLI (NNFI)
        if null_df > 0 and self.df > 0:
            self.tli = (null_chi / null_df - self.chi_square / self.df) / (
                null_chi / null_df - 1
            )
        else:
            self.tli = np.nan

        # RMSEA
        if self.df > 0:
            rmsea_val = max(self.chi_square - self.df, 0) / (self.df * (n - 1))
            self.rmsea = np.sqrt(rmsea_val)
        else:
            self.rmsea = 0.0

        # RMSEA 90% CI
        self.rmsea_ci = self._rmsea_ci()

        # SRMR
        self.srmr = self._compute_srmr()

    def _fit_null_model(self) -> tuple[float, int]:
        """Fit the independence (null) model: only variances, no covariances."""
        p = self._spec.n_obs
        obs_vars = self._spec.observed_vars

        # Independence model: each variable loads on itself, no relations
        model_str = "\n".join(f"{v} ~~ {v}" for v in obs_vars)
        tokens = parse_syntax(model_str)
        null_spec = build_specification(
            tokens, obs_vars, auto_var=False, auto_cov_latent=False
        )

        theta0 = null_spec.pack_start()

        from scipy import optimize as opt

        result = opt.minimize(
            ml_objective,
            theta0,
            args=(null_spec, self._sample_cov, self._n_obs),
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-6},
        )

        null_chi = (self._n_obs - 1) * result.fun
        null_df = p * (p + 1) // 2 - p  # only p variance parameters
        return null_chi, null_df

    def _rmsea_ci(self, alpha: float = 0.10) -> tuple[float, float]:
        """Compute RMSEA confidence interval using non-central chi-square."""
        n = self._n_obs
        df = self.df
        chi_sq = self.chi_square

        if df <= 0:
            return (0.0, 0.0)

        # Lower bound
        try:
            from scipy.optimize import brentq

            def lower_func(ncp):
                return stats.ncx2.cdf(chi_sq, df, ncp) - (1 - alpha / 2)

            # If cdf at ncp=0 exceeds target, a positive ncp is needed
            if stats.ncx2.cdf(chi_sq, df, 0) > (1 - alpha / 2):
                ncp_lower = brentq(lower_func, 0, max(chi_sq * 5, 200))
            else:
                ncp_lower = 0.0
            rmsea_lower = np.sqrt(max(ncp_lower, 0) / (df * (n - 1)))
        except (ValueError, RuntimeError):
            rmsea_lower = 0.0

        # Upper bound
        try:

            def upper_func(ncp):
                return stats.ncx2.cdf(chi_sq, df, ncp) - alpha / 2

            ncp_upper = brentq(upper_func, 0, max(chi_sq * 5, 200))
            rmsea_upper = np.sqrt(max(ncp_upper, 0) / (df * (n - 1)))
        except (ValueError, RuntimeError):
            rmsea_upper = np.nan

        return (rmsea_lower, rmsea_upper)

    def _compute_srmr(self) -> float:
        """Compute Standardized Root Mean Square Residual."""
        A, S_mat = self._spec.unpack(self._theta)
        sigma = _model_implied_cov(A, S_mat, self._spec.F)
        if sigma is None:
            return np.nan

        p = self._sample_cov.shape[0]
        residuals = self._sample_cov - sigma

        # Standardize
        d_inv = 1.0 / np.sqrt(np.diag(self._sample_cov))
        std_residuals = residuals * np.outer(d_inv, d_inv)

        # SRMR = sqrt(mean of squared elements in lower triangle including diagonal)
        mask = np.tril(np.ones((p, p), dtype=bool))
        srmr = np.sqrt(np.mean(std_residuals[mask] ** 2))
        return srmr

    def fit_indices(self) -> dict:
        """Return a dictionary of fit indices."""
        return {
            "chi_square": self.chi_square,
            "df": self.df,
            "p_value": self.p_value,
            "cfi": self.cfi,
            "tli": self.tli,
            "rmsea": self.rmsea,
            "rmsea_ci_lower": self.rmsea_ci[0],
            "rmsea_ci_upper": self.rmsea_ci[1],
            "srmr": self.srmr,
        }

    def estimates(self) -> pd.DataFrame:
        """Return parameter estimates as a DataFrame."""
        params = self._spec.params
        A_opt, S_opt = self._spec.unpack(self._theta)

        rows = []
        se_idx = 0  # index into SE vector (only free params)

        for p in params:
            if p.op == "=~":
                i = self._spec._idx(p.rhs)
                j = self._spec._idx(p.lhs)
                est = A_opt[i, j]
            elif p.op == "~":
                i = self._spec._idx(p.lhs)
                j = self._spec._idx(p.rhs)
                est = A_opt[i, j]
            elif p.op == "~~":
                i = self._spec._idx(p.lhs)
                j = self._spec._idx(p.rhs)
                est = S_opt[i, j]
            else:
                est = p.value

            if p.free:
                theta_idx = self._spec.param_theta_index(p)
                if theta_idx is not None and theta_idx < len(self._se):
                    se = self._se[theta_idx]
                else:
                    se = np.nan
                z = est / se if se > 0 and not np.isnan(se) else np.nan
                pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                ci_lower = est - 1.96 * se if not np.isnan(se) else np.nan
                ci_upper = est + 1.96 * se if not np.isnan(se) else np.nan
            else:
                se = np.nan
                z = np.nan
                pval = np.nan
                ci_lower = np.nan
                ci_upper = np.nan

            rows.append({
                "lhs": p.lhs,
                "op": p.op,
                "rhs": p.rhs,
                "est": est,
                "se": se,
                "z": z,
                "pvalue": pval,
                "ci.lower": ci_lower,
                "ci.upper": ci_upper,
                "free": p.free,
            })

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate a lavaan-style summary string."""
        lines = []
        lines.append("semla 0.1.0 — SEM Results")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Estimator                                           ML")
        lines.append(f"  Number of observations                         {self._n_obs:>6d}")
        lines.append("")

        # Fit indices
        lines.append("Model Test User Model:")
        lines.append("")
        lines.append(f"  Test statistic                           {self.chi_square:>12.3f}")
        lines.append(f"  Degrees of freedom                       {self.df:>12d}")
        lines.append(f"  P-value (Chi-square)                     {self.p_value:>12.3f}")
        lines.append("")

        lines.append("Fit Indices:")
        lines.append("")
        lines.append(f"  CFI                                      {self.cfi:>12.3f}")
        lines.append(f"  TLI (NNFI)                               {self.tli:>12.3f}")
        lines.append(f"  RMSEA                                    {self.rmsea:>12.3f}")
        lines.append(
            f"  90% CI RMSEA                       [{self.rmsea_ci[0]:>6.3f}, {self.rmsea_ci[1]:>6.3f}]"
        )
        lines.append(f"  SRMR                                     {self.srmr:>12.3f}")
        lines.append("")

        # Parameter estimates
        df = self.estimates()
        lines.append("Parameter Estimates:")
        lines.append("")

        # Group by operator
        for op, label in [("=~", "Latent Variables:"), ("~", "Regressions:"), ("~~", "Covariances/Variances:")]:
            subset = df[df["op"] == op]
            if subset.empty:
                continue

            lines.append(f"  {label}")

            current_lhs = None
            for _, row in subset.iterrows():
                if row["lhs"] != current_lhs:
                    current_lhs = row["lhs"]
                    lines.append(f"    {current_lhs} {op}")

                free_marker = "" if row["free"] else " (fixed)"
                se_str = f"{row['se']:.3f}" if not np.isnan(row["se"]) else ""
                z_str = f"{row['z']:.3f}" if not np.isnan(row["z"]) else ""
                p_str = f"{row['pvalue']:.3f}" if not np.isnan(row["pvalue"]) else ""

                lines.append(
                    f"      {row['rhs']:<15s} {row['est']:>8.3f}  {se_str:>8s}  {z_str:>8s}  {p_str:>8s}{free_marker}"
                )

            lines.append("")

        lines.append("=" * 60)

        output = "\n".join(lines)
        print(output)
        return output
