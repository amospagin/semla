"""Results container for multi-group SEM/CFA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import _model_implied_cov, ml_objective
from .multigroup import (
    MultiGroupEstimationResult,
    MultiGroupSpec,
    _multigroup_compute_se,
    multigroup_ml_objective,
)
from .specification import build_specification
from .syntax import parse_syntax


class MultiGroupResults:
    """Container for multi-group model results."""

    def __init__(self, est: MultiGroupEstimationResult):
        self._est = est
        self._mg = est.mg_spec
        self._theta = est.theta_combined

        # Compute SEs
        self._se = _multigroup_compute_se(self._theta, self._mg)

        # Fit indices
        self._compute_fit_indices()

    @property
    def converged(self) -> bool:
        return self._est.converged

    def _compute_fit_indices(self):
        n_groups = len(self._mg.group_names)
        p = self._mg.group_specs[0].n_obs

        # Total chi-square: sum of per-group chi-squares
        self.chi_square = 0.0
        for g in range(n_groups):
            theta_g = self._theta[self._mg.theta_mapping[g]]
            f_g = ml_objective(
                theta_g,
                self._mg.group_specs[g],
                self._mg.group_sample_covs[g],
                self._mg.group_n_obs[g],
            )
            self.chi_square += (self._mg.group_n_obs[g] - 1) * f_g

        # Degrees of freedom
        total_data_points = n_groups * (p * (p + 1) // 2)
        self.df = total_data_points - self._mg.n_free_combined

        # P-value
        self.p_value = (
            1.0 - stats.chi2.cdf(self.chi_square, self.df)
            if self.df > 0
            else np.nan
        )

        # Null model
        null_chi, null_df = self._fit_null_model()

        # CFI
        num = max(self.chi_square - self.df, 0)
        den = max(null_chi - null_df, 0)
        self.cfi = 1.0 - num / den if den > 0 else 1.0

        # TLI
        if null_df > 0 and self.df > 0:
            self.tli = (null_chi / null_df - self.chi_square / self.df) / (
                null_chi / null_df - 1
            )
        else:
            self.tli = np.nan

        # RMSEA
        N = self._mg.n_total
        if self.df > 0:
            rmsea_val = max(self.chi_square - self.df, 0) / (self.df * (N - n_groups))
            self.rmsea = np.sqrt(rmsea_val)
        else:
            self.rmsea = 0.0

        # SRMR (average across groups)
        self.srmr = self._compute_srmr()

    def _fit_null_model(self) -> tuple[float, int]:
        """Fit null (independence) model for each group."""
        n_groups = len(self._mg.group_names)
        p = self._mg.group_specs[0].n_obs
        obs_vars = self._mg.group_specs[0].observed_vars

        null_chi = 0.0
        for g in range(n_groups):
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
                args=(null_spec, self._mg.group_sample_covs[g], self._mg.group_n_obs[g]),
                method="BFGS",
                options={"maxiter": 1000, "gtol": 1e-6},
            )
            null_chi += (self._mg.group_n_obs[g] - 1) * result.fun

        null_df = n_groups * (p * (p + 1) // 2 - p)
        return null_chi, null_df

    def _compute_srmr(self) -> float:
        """Average SRMR across groups."""
        srmr_sum = 0.0
        n_groups = len(self._mg.group_names)

        for g in range(n_groups):
            theta_g = self._theta[self._mg.theta_mapping[g]]
            A_g, S_g = self._mg.group_specs[g].unpack(theta_g)
            sigma_g = _model_implied_cov(A_g, S_g, self._mg.group_specs[g].F)
            if sigma_g is None:
                return np.nan

            S_sample = self._mg.group_sample_covs[g]
            p = S_sample.shape[0]
            residuals = S_sample - sigma_g
            d_inv = 1.0 / np.sqrt(np.diag(S_sample))
            std_resid = residuals * np.outer(d_inv, d_inv)
            mask = np.tril(np.ones((p, p), dtype=bool))
            srmr_sum += np.sqrt(np.mean(std_resid[mask] ** 2))

        return srmr_sum / n_groups

    def fit_indices(self) -> dict:
        return {
            "chi_square": self.chi_square,
            "df": self.df,
            "p_value": self.p_value,
            "cfi": self.cfi,
            "tli": self.tli,
            "rmsea": self.rmsea,
            "srmr": self.srmr,
            "n_groups": len(self._mg.group_names),
            "invariance": self._mg.invariance,
        }

    def estimates(self) -> pd.DataFrame:
        """Return parameter estimates with group column."""
        all_rows = []
        n_groups = len(self._mg.group_names)

        for g in range(n_groups):
            spec_g = self._mg.group_specs[g]
            mapping_g = self._mg.theta_mapping[g]
            theta_g = self._theta[mapping_g]
            A_opt, S_opt = spec_g.unpack(theta_g)

            for p in spec_g.params:
                if p.op == "=~":
                    i = spec_g._idx(p.rhs)
                    j = spec_g._idx(p.lhs)
                    est = A_opt[i, j]
                elif p.op == "~":
                    i = spec_g._idx(p.lhs)
                    j = spec_g._idx(p.rhs)
                    est = A_opt[i, j]
                elif p.op == "~~":
                    i = spec_g._idx(p.lhs)
                    j = spec_g._idx(p.rhs)
                    est = S_opt[i, j]
                else:
                    est = p.value

                if p.free:
                    theta_idx = spec_g.param_theta_index(p)
                    if theta_idx is not None:
                        combined_idx = mapping_g[theta_idx]
                        se = self._se[combined_idx] if combined_idx < len(self._se) else np.nan
                    else:
                        se = np.nan
                    z = est / se if se > 0 and not np.isnan(se) else np.nan
                    pval = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                else:
                    se = np.nan
                    z = np.nan
                    pval = np.nan

                all_rows.append({
                    "group": self._mg.group_names[g],
                    "lhs": p.lhs,
                    "op": p.op,
                    "rhs": p.rhs,
                    "est": est,
                    "se": se,
                    "z": z,
                    "pvalue": pval,
                    "free": p.free,
                })

        return pd.DataFrame(all_rows)

    def summary(self) -> str:
        lines = []
        n_groups = len(self._mg.group_names)
        lines.append("semla 0.1.0 — Multi-Group SEM Results")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Estimator                                           ML")
        lines.append(f"  Number of groups                               {n_groups:>6d}")
        lines.append(f"  Total observations                         {self._mg.n_total:>6d}")
        for g, gname in enumerate(self._mg.group_names):
            lines.append(f"    Group {gname:<20s}                  {self._mg.group_n_obs[g]:>6d}")
        lines.append(f"  Invariance                          {self._mg.invariance:>12s}")
        lines.append(f"  Free parameters                            {self._mg.n_free_combined:>6d}")
        lines.append("")

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
        lines.append(f"  SRMR                                     {self.srmr:>12.3f}")
        lines.append("")

        # Per-group estimates
        df = self.estimates()
        for g, gname in enumerate(self._mg.group_names):
            lines.append(f"Group: {gname}")
            lines.append("-" * 60)
            gdf = df[df["group"] == gname]

            for op, label in [("=~", "Latent Variables:"), ("~", "Regressions:"), ("~~", "Covariances/Variances:")]:
                subset = gdf[gdf["op"] == op]
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
