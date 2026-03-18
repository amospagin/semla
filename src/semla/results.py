"""Fit indices, parameter tables, and lavaan-style summary output."""

from __future__ import annotations

import warnings

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
        self._estimator = getattr(est_result, "estimator_type", "ML")

        # Compute standard errors
        if self._estimator == "DWLS":
            from .dwls import _compute_se_dwls
            self._se = _compute_se_dwls(
                self._theta, self._spec,
                est_result.polychoric_cov,
                est_result.weight_diagonal,
                est_result.gamma_diagonal,
                self._n_obs,
            )
        else:
            self._se = _compute_se(
                self._theta, self._spec, self._sample_cov, self._n_obs
            )

        # Compute fit indices
        self._compute_fit_indices()

        # Check for Heywood cases
        self._check_heywood()

    @property
    def converged(self) -> bool:
        return self._est.converged

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_free(self) -> int:
        return self._spec.n_free

    # --- Post-estimation checks ---

    def _check_heywood(self):
        """Warn if any variance estimates are negative (Heywood cases)."""
        A_opt, S_opt = self._spec.unpack(self._theta)
        heywood_vars = []
        for p in self._spec.params:
            if p.op == "~~" and p.lhs == p.rhs and p.free:
                i = self._spec._idx(p.lhs)
                if S_opt[i, i] < 0:
                    heywood_vars.append((p.lhs, S_opt[i, i]))

        if heywood_vars:
            var_list = ", ".join(
                f"{name} ({val:.4f})" for name, val in heywood_vars
            )
            warnings.warn(
                f"Negative variance estimate(s) detected (Heywood case): "
                f"{var_list}. This may indicate model misspecification, "
                f"too few observations, or empirical underidentification.",
                RuntimeWarning,
                stacklevel=4,
            )

    # --- Fit indices ---

    def _compute_fit_indices(self):
        p = self._spec.n_obs
        n = self._n_obs

        # Chi-square
        self.fmin = self._est.fmin
        self.df = p * (p + 1) // 2 - self._spec.n_free

        if self._estimator == "DWLS":
            from .dwls import _scaled_chi_square
            self.chi_square, self._scaling_factor = _scaled_chi_square(
                self._theta, self._spec, self._est.polychoric_cov,
                self._est.gamma_diagonal, n, self.df,
            )
        else:
            self.chi_square = (n - 1) * self.fmin

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

        # Information criteria
        # AIC = chi_square + 2*k, BIC = chi_square + k*log(n)
        k = self._spec.n_free
        self.aic = self.chi_square + 2 * k
        self.bic = self.chi_square + k * np.log(n)
        # Sample-size adjusted BIC (Sclove 1987)
        self.abic = self.chi_square + k * np.log((n + 2) / 24)

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
            "aic": self.aic,
            "bic": self.bic,
            "abic": self.abic,
        }

    def r_squared(self) -> dict:
        """Compute R-squared for endogenous variables.

        R² = 1 - (residual variance / total implied variance)

        Returns
        -------
        dict
            Variable name -> R² value for each endogenous variable.
        """
        A_opt, S_opt = self._spec.unpack(self._theta)

        # Total implied covariance (all variables including latent)
        n = self._spec.n_vars
        I_mat = np.eye(n)
        IminA_inv = np.linalg.inv(I_mat - A_opt)
        total_cov = IminA_inv @ S_opt @ IminA_inv.T

        result = {}
        for p in self._spec.params:
            if p.op == "~~" and p.lhs == p.rhs and p.free:
                i = self._spec._idx(p.lhs)
                total_var = total_cov[i, i]
                resid_var = S_opt[i, i]
                if total_var > 0:
                    r2 = 1.0 - resid_var / total_var
                    # Only report for endogenous variables (those with incoming paths)
                    has_incoming = np.any(A_opt[i, :] != 0)
                    if has_incoming:
                        result[p.lhs] = max(r2, 0.0)

        return result

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

    def standardized_estimates(self, type: str = "std.all") -> pd.DataFrame:
        """Return standardized parameter estimates.

        Parameters
        ----------
        type : str
            Type of standardization:
            - ``"std.all"`` — fully standardized (by both LV and observed SD)
            - ``"std.lv"`` — standardized by latent variable SD only

        Returns
        -------
        pd.DataFrame
            Same structure as ``estimates()`` with added ``est.std`` column.
        """
        if type not in ("std.all", "std.lv"):
            raise ValueError(f"type must be 'std.all' or 'std.lv', got '{type}'")

        df = self.estimates()
        A_opt, S_opt = self._spec.unpack(self._theta)

        # Compute implied total covariance for all variables (including latent)
        n = self._spec.n_vars
        I = np.eye(n)
        IminA_inv = np.linalg.inv(I - A_opt)
        total_cov = IminA_inv @ S_opt @ IminA_inv.T

        # SD of each variable (sqrt of diagonal of total covariance)
        total_sd = np.sqrt(np.maximum(np.diag(total_cov), 0))

        std_est = []
        for _, row in df.iterrows():
            lhs, op, rhs, est = row["lhs"], row["op"], row["rhs"], row["est"]

            if op == "=~":
                # Loading: lhs is latent, rhs is indicator
                sd_lv = total_sd[self._spec._idx(lhs)]
                sd_ov = total_sd[self._spec._idx(rhs)]

                if type == "std.lv":
                    std_val = est * sd_lv if sd_lv > 0 else np.nan
                else:  # std.all
                    std_val = est * sd_lv / sd_ov if sd_lv > 0 and sd_ov > 0 else np.nan

            elif op == "~":
                # Regression: lhs ~ rhs
                sd_iv = total_sd[self._spec._idx(rhs)]
                sd_dv = total_sd[self._spec._idx(lhs)]

                if type == "std.lv":
                    # Only standardize if both are latent
                    if rhs in self._spec.latent_vars and lhs in self._spec.latent_vars:
                        std_val = est * sd_iv / sd_dv if sd_dv > 0 else np.nan
                    else:
                        std_val = est
                else:  # std.all
                    std_val = est * sd_iv / sd_dv if sd_iv > 0 and sd_dv > 0 else np.nan

            elif op == "~~":
                i = self._spec._idx(lhs)
                j = self._spec._idx(rhs)
                sd_i = total_sd[i]
                sd_j = total_sd[j]

                if type == "std.lv":
                    if lhs in self._spec.latent_vars and rhs in self._spec.latent_vars:
                        std_val = est / (sd_i * sd_j) if sd_i > 0 and sd_j > 0 else np.nan
                    elif lhs in self._spec.latent_vars or rhs in self._spec.latent_vars:
                        std_val = est
                    else:
                        std_val = est
                else:  # std.all
                    std_val = est / (sd_i * sd_j) if sd_i > 0 and sd_j > 0 else np.nan
            else:
                std_val = est

            std_est.append(std_val)

        df["est.std"] = std_est
        return df

    def modindices(self, min_mi: float = 0.0, sort: bool = True) -> pd.DataFrame:
        """Compute modification indices for fixed parameters.

        A modification index (MI) approximates the expected drop in
        chi-square (df=1) if a currently fixed-to-zero parameter were freed.
        Uses the univariate score (Lagrange multiplier) test:

            MI = (N-1) * g_r^2 / J_rr

        where g_r = dF_ML/dtheta_r and J_rr = 0.5*tr(Sigma^{-1} dSigma_r Sigma^{-1} dSigma_r).

        Parameters
        ----------
        min_mi : float
            Only return parameters with MI >= this value.
        sort : bool
            Sort by MI descending.

        Returns
        -------
        pd.DataFrame
            Columns: lhs, op, rhs, mi, epc (expected parameter change).
        """
        A_opt, S_opt = self._spec.unpack(self._theta)
        n = self._spec.n_vars
        N = self._n_obs

        # Model-implied covariance
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        sigma_inv = np.linalg.inv(sigma)

        # W matrix for score: dF/dtheta = tr(W @ dSigma)
        W = sigma_inv - sigma_inv @ self._sample_cov @ sigma_inv

        eps = 1e-7
        rows = []

        def _compute_mi(dSigma_r, symmetric=False):
            """Compute MI for a candidate fixed parameter."""
            # g_r = dF_ML/dtheta_r
            g_r = np.trace(W @ dSigma_r)

            # J_rr = expected information element
            # For symmetric (S) off-diagonal params, use 0.5*tr (parameter appears twice)
            # For asymmetric (A) params, use tr (parameter appears once)
            SinvdS = sigma_inv @ dSigma_r
            J_rr = np.trace(SinvdS @ SinvdS)
            if symmetric:
                J_rr *= 0.5

            if J_rr > 1e-15:
                mi = (N - 1) * g_r ** 2 / J_rr
                epc = g_r / J_rr
            else:
                mi = 0.0
                epc = 0.0
            return mi, epc

        # --- Check all zero-fixed cells in A (loadings/regressions) ---
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._spec.A_free[i, j] or A_opt[i, j] != 0.0:
                    continue

                var_i = self._spec.all_vars[i]
                var_j = self._spec.all_vars[j]

                if var_j in self._spec.latent_vars and var_i not in self._spec.latent_vars:
                    op, lhs, rhs = "=~", var_j, var_i
                elif var_j in self._spec.latent_vars and var_i in self._spec.latent_vars:
                    op, lhs, rhs = "~", var_i, var_j
                else:
                    op, lhs, rhs = "~", var_i, var_j

                A_plus = A_opt.copy()
                A_plus[i, j] = eps
                sig_plus = _model_implied_cov(A_plus, S_opt, self._spec.F)
                A_minus = A_opt.copy()
                A_minus[i, j] = -eps
                sig_minus = _model_implied_cov(A_minus, S_opt, self._spec.F)
                if sig_plus is None or sig_minus is None:
                    continue

                dSigma_r = (sig_plus - sig_minus) / (2 * eps)
                mi, epc = _compute_mi(dSigma_r)

                if mi >= min_mi:
                    rows.append({"lhs": lhs, "op": op, "rhs": rhs, "mi": mi, "epc": epc})

        # --- Check all zero-fixed cells in S (covariances) ---
        # Only between observed variables (residual covariances)
        obs_set = set(self._spec.observed_vars)
        for i in range(n):
            for j in range(i + 1):
                if self._spec.S_free[i, j] or S_opt[i, j] != 0.0:
                    continue

                var_i = self._spec.all_vars[i]
                var_j = self._spec.all_vars[j]

                # Skip latent-observed and latent-latent covariances
                if var_i not in obs_set or var_j not in obs_set:
                    continue

                S_plus = S_opt.copy()
                S_plus[i, j] += eps
                S_plus[j, i] += eps
                sig_plus = _model_implied_cov(A_opt, S_plus, self._spec.F)
                S_minus = S_opt.copy()
                S_minus[i, j] -= eps
                S_minus[j, i] -= eps
                sig_minus = _model_implied_cov(A_opt, S_minus, self._spec.F)
                if sig_plus is None or sig_minus is None:
                    continue

                dSigma_r = (sig_plus - sig_minus) / (2 * eps)
                is_offdiag = (i != j)
                mi, epc = _compute_mi(dSigma_r, symmetric=is_offdiag)

                if mi >= min_mi:
                    rows.append({"lhs": var_i, "op": "~~", "rhs": var_j, "mi": mi, "epc": epc})

        df = pd.DataFrame(rows, columns=["lhs", "op", "rhs", "mi", "epc"])
        if sort and len(df) > 0:
            df = df.sort_values("mi", ascending=False).reset_index(drop=True)
        return df

    def summary(self) -> str:
        """Generate a lavaan-style summary string."""
        lines = []
        lines.append("semla 0.1.0 — SEM Results")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Estimator                                         {self._estimator:>4s}")
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
        lines.append("Information Criteria:")
        lines.append("")
        lines.append(f"  AIC                                      {self.aic:>12.3f}")
        lines.append(f"  BIC                                      {self.bic:>12.3f}")
        lines.append(f"  Adjusted BIC                             {self.abic:>12.3f}")
        lines.append("")

        # R-squared
        r2 = self.r_squared()
        if r2:
            lines.append("R-Square:")
            lines.append("")
            for var, val in r2.items():
                lines.append(f"  {var:<40s} {val:>12.3f}")
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
