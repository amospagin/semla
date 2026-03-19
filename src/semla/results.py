"""Fit indices, parameter tables, and lavaan-style summary output."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import EstimationResult, _compute_se, _model_implied_cov, _model_implied_mean, ml_objective
from .specification import ModelSpecification, build_specification
from .syntax import FormulaToken, parse_syntax


class ModelResults:
    """Container for estimated model results and fit statistics."""

    def __init__(self, est_result: EstimationResult, defined_params=None):
        self._est = est_result
        self._spec = est_result.spec
        self._theta = est_result.theta
        self._sample_cov = est_result.sample_cov
        self._n_obs = est_result.n_obs
        self._estimator = getattr(est_result, "_estimator_type",
                                   getattr(est_result, "estimator_type", "ML"))
        self._defined_params = defined_params or []

        # Compute standard errors
        self._vcov = None  # set by ML path; other estimators compute on demand
        if self._estimator == "DWLS":
            from .dwls import _compute_se_dwls
            self._se = _compute_se_dwls(
                self._theta, self._spec,
                est_result.polychoric_cov,
                est_result.weight_diagonal,
                est_result.gamma_diagonal,
                self._n_obs,
            )
        elif self._estimator == "MLR":
            from .robust import compute_gamma, compute_robust_se
            # Use ML sample covariance (/ N) for Gamma, matching lavaan
            n = self._n_obs
            sample_cov_ml = self._sample_cov * (n - 1) / n
            self._gamma = compute_gamma(est_result.raw_data, sample_cov_ml)
            self._se = compute_robust_se(
                self._theta, self._spec, self._sample_cov, self._n_obs,
                self._gamma, raw_data=est_result.raw_data,
            )
        elif getattr(est_result, "_missing_method", None) == "fiml":
            from .fiml import _compute_se_fiml
            self._se = _compute_se_fiml(
                self._theta, self._spec,
                est_result._pattern_groups, self._n_obs,
            )
        else:
            self._se, self._vcov = _compute_se(
                self._theta, self._spec, self._sample_cov, self._n_obs,
                return_vcov=True,
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
        data_points = p * (p + 1) // 2
        if self._spec.meanstructure:
            data_points += p  # add mean information
        self.df = data_points - self._spec.n_free

        if getattr(self._est, "_missing_method", None) == "fiml":
            self.chi_square = self._est._fiml_chi_square
        elif self._estimator == "DWLS":
            from .dwls import _scaled_chi_square
            self.chi_square, self._scaling_factor = _scaled_chi_square(
                self._theta, self._spec, self._est.polychoric_cov,
                self._est.gamma_diagonal, n, self.df,
            )
        elif self._estimator == "MLR":
            from .robust import satorra_bentler_chi_square
            self.chi_square, self._scaling_factor = satorra_bentler_chi_square(
                self.fmin, n, self.df, self._theta, self._spec,
                self._sample_cov, self._gamma,
                raw_data=self._est.raw_data,
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

    def _get_label_maps(self) -> tuple[dict, dict]:
        """Build label -> value and label -> SE mappings from estimates."""
        label_values = {}
        label_se = {}
        for p in self._spec.params:
            if p.free and p.label:
                theta_idx = self._spec.param_theta_index(p)
                if theta_idx is None:
                    continue

                # Get the estimated value
                if p.op == "=~":
                    A, S = self._spec.unpack(self._theta)
                    i = self._spec._idx(p.rhs)
                    j = self._spec._idx(p.lhs)
                    label_values[p.label] = A[i, j]
                elif p.op == "~":
                    A, S = self._spec.unpack(self._theta)
                    i = self._spec._idx(p.lhs)
                    j = self._spec._idx(p.rhs)
                    label_values[p.label] = A[i, j]
                elif p.op == "~~":
                    A, S = self._spec.unpack(self._theta)
                    i = self._spec._idx(p.lhs)
                    j = self._spec._idx(p.rhs)
                    label_values[p.label] = S[i, j]

                if theta_idx < len(self._se):
                    label_se[p.label] = self._se[theta_idx]
        return label_values, label_se

    def defined_estimates(self) -> pd.DataFrame:
        """Return estimates for user-defined parameters (:= operator).

        Returns
        -------
        pd.DataFrame
            Columns: name, expression, est, se, z, pvalue.
            Empty DataFrame if no defined parameters.
        """
        if not self._defined_params:
            return pd.DataFrame(columns=["name", "expression", "est", "se", "z", "pvalue"])

        from .defined import evaluate_defined_params, compute_defined_se

        label_values, label_se = self._get_label_maps()

        # Evaluate expressions
        results = evaluate_defined_params(self._defined_params, label_values)

        # Compute SEs via delta method
        ses = compute_defined_se(self._defined_params, label_values, label_se)

        rows = []
        for i, res in enumerate(results):
            se = ses[i]
            est = res["est"]
            z = est / se if se > 0 and not np.isnan(se) else np.nan
            pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            rows.append({
                "name": res["name"],
                "expression": res["expression"],
                "est": est,
                "se": se,
                "z": z,
                "pvalue": pval,
            })

        return pd.DataFrame(rows)

    def fitted(self) -> dict:
        """Return model-implied moments (covariance matrix and mean vector).

        Returns
        -------
        dict
            ``"cov"`` : pd.DataFrame — model-implied covariance matrix.
            ``"mean"`` : pd.Series or None — model-implied mean vector
            (only when meanstructure=True).
        """
        obs = self._spec.observed_vars
        A_opt, S_opt = self._spec.unpack(self._theta)
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        if sigma is None:
            raise RuntimeError("Cannot compute model-implied covariance.")

        result = {"cov": pd.DataFrame(sigma, index=obs, columns=obs)}

        if self._spec.meanstructure and self._spec.m_values is not None:
            m_est = self._spec.unpack_m(self._theta)
            mu = _model_implied_mean(A_opt, m_est, self._spec.F)
            result["mean"] = pd.Series(mu, index=obs) if mu is not None else None
        else:
            result["mean"] = None

        return result

    def vcov(self) -> pd.DataFrame:
        """Return the parameter variance-covariance matrix.

        This is the inverse of the expected information matrix, with rows
        and columns labeled by parameter names (lhs op rhs).

        Returns
        -------
        pd.DataFrame
            Square matrix of parameter covariances.
        """
        if self._vcov is None:
            # Compute on demand for non-ML estimators
            _, self._vcov = _compute_se(
                self._theta, self._spec, self._sample_cov, self._n_obs,
                return_vcov=True,
            )

        # Build labels from free parameters
        est_df = self.estimates()
        free_params = est_df[est_df["free"]]
        labels = [f"{r['lhs']} {r['op']} {r['rhs']}" for _, r in free_params.iterrows()]

        n = self._vcov.shape[0]
        if len(labels) != n:
            # Fallback: use numeric labels
            labels = [f"theta_{i}" for i in range(n)]

        return pd.DataFrame(self._vcov, index=labels, columns=labels)

    def residuals(self, type: str = "raw") -> np.ndarray:
        """Return residual covariance matrix (observed - implied).

        Parameters
        ----------
        type : str
            ``"raw"`` or ``"standardized"`` (normalized by observed SD).
        """
        A_opt, S_opt = self._spec.unpack(self._theta)
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        if sigma is None:
            return np.full_like(self._sample_cov, np.nan)

        resid = self._sample_cov - sigma
        if type == "standardized":
            d = np.sqrt(np.diag(self._sample_cov))
            d = np.where(d > 0, d, 1.0)
            resid = resid / np.outer(d, d)
        elif type != "raw":
            raise ValueError(f"type must be 'raw' or 'standardized', got '{type}'")
        return resid

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

    def reliability(self) -> dict:
        """Compute reliability measures for each latent factor.

        Returns
        -------
        dict
            Factor name -> dict with 'omega' (McDonald's omega) and
            'alpha' (Cronbach's alpha from model-implied covariance).

        Notes
        -----
        McDonald's omega = (sum of loadings)^2 / ((sum of loadings)^2 + sum of residual variances)
        Cronbach's alpha = (k / (k-1)) * (1 - sum(diag(Sigma_f)) / sum(Sigma_f))
        where Sigma_f is the model-implied covariance for the factor's indicators.
        """
        A_opt, S_opt = self._spec.unpack(self._theta)

        # Group indicators by factor
        factor_indicators: dict[str, list[str]] = {}
        for p in self._spec.params:
            if p.op == "=~":
                factor_indicators.setdefault(p.lhs, []).append(p.rhs)

        # Model-implied covariance of observed variables
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        if sigma is None:
            return {}

        result = {}
        for lv, indicators in factor_indicators.items():
            if len(indicators) < 2:
                continue

            # Get loadings and residual variances for this factor
            lv_idx = self._spec._idx(lv)
            lv_var = S_opt[lv_idx, lv_idx]  # latent variance (in S, not total)

            loadings = []
            resid_vars = []
            obs_indices = []
            for ind in indicators:
                i = self._spec._idx(ind)
                loading = A_opt[i, lv_idx]
                loadings.append(loading)
                resid_vars.append(S_opt[i, i])
                obs_indices.append(self._spec.observed_vars.index(ind))

            loadings = np.array(loadings)
            resid_vars = np.array(resid_vars)

            # McDonald's omega
            sum_loadings = np.sum(loadings)
            omega_num = (sum_loadings ** 2) * lv_var
            omega_den = omega_num + np.sum(resid_vars)
            omega = omega_num / omega_den if omega_den > 0 else np.nan

            # Cronbach's alpha from model-implied covariance
            k = len(indicators)
            idx = np.array(obs_indices)
            Sigma_f = sigma[np.ix_(idx, idx)]
            sum_all = np.sum(Sigma_f)
            sum_diag = np.sum(np.diag(Sigma_f))
            alpha = (k / (k - 1)) * (1 - sum_diag / sum_all) if sum_all > 0 else np.nan

            result[lv] = {"omega": omega, "alpha": alpha}

        return result

    def factor_scores(self, data: pd.DataFrame, method: str = "regression") -> pd.DataFrame:
        """Predict latent variable scores for each observation.

        Parameters
        ----------
        data : pd.DataFrame
            Data with observed variables.
        method : str
            ``"regression"`` (Thurstone/Thomson) or ``"bartlett"``.

        Returns
        -------
        pd.DataFrame
            One column per latent variable.
        """
        A_opt, S_opt = self._spec.unpack(self._theta)
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        if sigma is None:
            raise RuntimeError("Cannot compute factor scores: model-implied covariance is invalid.")

        obs_data = data[self._spec.observed_vars].values
        sigma_inv = np.linalg.inv(sigma)

        # Full implied covariance
        n = self._spec.n_vars
        I_mat = np.eye(n)
        IminA_inv = np.linalg.inv(I_mat - A_opt)
        full_cov = IminA_inv @ S_opt @ IminA_inv.T

        # Covariance between latent and observed: C_eta_y = full_cov[latent, :] @ F'
        latent_indices = [self._spec._idx(lv) for lv in self._spec.latent_vars]
        obs_indices = list(range(self._spec.n_obs))

        # C(eta, y) = full_cov[latent_idx, :][:, obs_idx_in_full]
        # where obs_idx_in_full = observed variable indices in all_vars
        obs_in_all = [self._spec._idx(v) for v in self._spec.observed_vars]
        C_eta_y = full_cov[np.ix_(latent_indices, obs_in_all)]

        if method == "regression":
            # Regression scores: eta_hat = C_eta_y @ Sigma^{-1} @ (y - mu)'
            W = C_eta_y @ sigma_inv

            # Center data
            if self._spec.meanstructure and self._est.sample_mean is not None:
                centered = obs_data - self._est.sample_mean
            else:
                centered = obs_data - obs_data.mean(axis=0)

            scores = centered @ W.T

        elif method == "bartlett":
            # Bartlett scores: weighted by inverse residual variance
            # Lambda = loading matrix (obs x latent)
            n_obs = self._spec.n_obs
            n_lat = self._spec.n_latent
            Lambda = np.zeros((n_obs, n_lat))
            for i, ov in enumerate(self._spec.observed_vars):
                ov_idx = self._spec._idx(ov)
                for j, lv in enumerate(self._spec.latent_vars):
                    lv_idx = self._spec._idx(lv)
                    Lambda[i, j] = A_opt[ov_idx, lv_idx]

            # Theta = diagonal residual covariance
            Theta_inv = np.diag([1.0 / max(S_opt[self._spec._idx(v), self._spec._idx(v)], 1e-10)
                                 for v in self._spec.observed_vars])

            # Bartlett: (Lambda' Theta^{-1} Lambda)^{-1} Lambda' Theta^{-1} (y - mu)
            LtTinv = Lambda.T @ Theta_inv
            W = np.linalg.solve(LtTinv @ Lambda, LtTinv)

            if self._spec.meanstructure and self._est.sample_mean is not None:
                centered = obs_data - self._est.sample_mean
            else:
                centered = obs_data - obs_data.mean(axis=0)

            scores = centered @ W.T
        else:
            raise ValueError(f"method must be 'regression' or 'bartlett', got '{method}'")

        return pd.DataFrame(scores, columns=self._spec.latent_vars, index=data.index)

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
            elif p.op == "~1":
                if self._spec.meanstructure:
                    m = self._spec.unpack_m(self._theta)
                    i = self._spec._idx(p.lhs)
                    est = m[i]
                else:
                    est = p.value
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

            MI = score_r^2 / I_rr

        where score_r = -(N-1)/2 * tr(W @ dSigma_r) is the score and
        I_rr = (N-1)/2 * tr(Sigma^{-1} dSigma_r Sigma^{-1} dSigma_r) is
        the expected information element.

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
        k = len(self._theta)  # number of free parameters

        # Model-implied covariance
        sigma = _model_implied_cov(A_opt, S_opt, self._spec.F)
        sigma_inv = np.linalg.inv(sigma)

        # W matrix for score: dF/dtheta = tr(W @ dSigma)
        W = sigma_inv - sigma_inv @ self._sample_cov @ sigma_inv

        eps = 1e-7
        p = sigma.shape[0]

        # Precompute dSigma for all free parameters and Σ⁻¹ @ dΣ_i
        dSigma_free = np.zeros((k, p, p))
        SinvdS_free = np.zeros((k, p, p))
        for idx in range(k):
            theta_plus = self._theta.copy()
            theta_minus = self._theta.copy()
            theta_plus[idx] += eps
            theta_minus[idx] -= eps
            A_p, S_p = self._spec.unpack(theta_plus)
            A_m, S_m = self._spec.unpack(theta_minus)
            sig_p = _model_implied_cov(A_p, S_p, self._spec.F)
            sig_m = _model_implied_cov(A_m, S_m, self._spec.F)
            if sig_p is not None and sig_m is not None:
                dSigma_free[idx] = (sig_p - sig_m) / (2 * eps)
                SinvdS_free[idx] = sigma_inv @ dSigma_free[idx]

        # I_11 = expected information for free parameters
        I_11 = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                val = 0.5 * (N - 1) * np.trace(SinvdS_free[i] @ SinvdS_free[j])
                I_11[i, j] = val
                I_11[j, i] = val

        # I_11^{-1} for Schur complement
        try:
            I_11_inv = np.linalg.inv(I_11)
        except np.linalg.LinAlgError:
            I_11_inv = np.linalg.pinv(I_11)

        rows = []

        def _compute_mi(dSigma_r):
            """Compute MI using the Schur complement of the augmented information.

            MI = score_r² / V_rr where V_rr = I_rr - I_r1 @ I_11^{-1} @ I_1r
            accounts for correlation between the candidate and free parameters.
            """
            # Score for this parameter
            g_r = np.trace(W @ dSigma_r)
            score_r = -0.5 * (N - 1) * g_r

            # I_rr: expected information for candidate parameter
            SinvdS_r = sigma_inv @ dSigma_r
            I_rr = 0.5 * (N - 1) * np.trace(SinvdS_r @ SinvdS_r)

            # I_r1: cross-information with free parameters
            I_r1 = np.zeros(k)
            for idx in range(k):
                I_r1[idx] = 0.5 * (N - 1) * np.trace(SinvdS_r @ SinvdS_free[idx])

            # Schur complement: V_rr = I_rr - I_r1' I_11^{-1} I_r1
            V_rr = I_rr - I_r1 @ I_11_inv @ I_r1

            if V_rr > 1e-15:
                mi = score_r ** 2 / V_rr
                epc = score_r / V_rr
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
                    # Skip regressions between latent variables that already
                    # have a covariance — lavaan excludes these by default
                    idx_i = self._spec._idx(var_i)
                    idx_j = self._spec._idx(var_j)
                    if self._spec.S_free[idx_i, idx_j] or S_opt[idx_i, idx_j] != 0:
                        continue
                    op, lhs, rhs = "~", var_i, var_j
                else:
                    # Skip regressions between observed variables —
                    # lavaan excludes these by default
                    continue

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
                mi, epc = _compute_mi(dSigma_r)

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
        for op, label in [("=~", "Latent Variables:"), ("~", "Regressions:"), ("~1", "Intercepts:"), ("~~", "Covariances/Variances:")]:
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

        # Defined parameters
        defined_df = self.defined_estimates()
        if len(defined_df) > 0:
            lines.append("  Defined Parameters:")
            for _, row in defined_df.iterrows():
                se_str = f"{row['se']:.3f}" if not np.isnan(row["se"]) else ""
                z_str = f"{row['z']:.3f}" if not np.isnan(row["z"]) else ""
                p_str = f"{row['pvalue']:.3f}" if not np.isnan(row["pvalue"]) else ""
                lines.append(
                    f"    {row['name']:<17s} {row['est']:>8.3f}  {se_str:>8s}  {z_str:>8s}  {p_str:>8s}"
                )
            lines.append("")

        lines.append("=" * 60)

        output = "\n".join(lines)
        print(output)
        return output
