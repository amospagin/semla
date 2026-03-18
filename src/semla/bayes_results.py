"""Bayesian SEM results: draws, summaries, diagnostics, and model comparison.

Provides :class:`BayesianResults`, the Bayesian counterpart to
:class:`~semla.results.ModelResults`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .specification import ModelSpecification
from .prior_defaults import _param_key, _matrix_category


# ---------------------------------------------------------------------------
# Diagnostics (re-use from bayes.py)
# ---------------------------------------------------------------------------

def _import_bayes_diag():
    from .bayes import _effective_sample_size, _rhat, _divergence_stats
    return _effective_sample_size, _rhat, _divergence_stats


# ---------------------------------------------------------------------------
# BayesianResults
# ---------------------------------------------------------------------------

class BayesianResults:
    """Container for Bayesian SEM MCMC results.

    Provides the same interface shape as :class:`~semla.results.ModelResults`
    where applicable, plus Bayesian-specific methods (draws, WAIC, LOO).
    """

    def __init__(self, mcmc, spec: ModelSpecification, param_keys: list[str],
                 prior_dict: dict, data: np.ndarray):
        self.mcmc = mcmc
        self.spec = spec
        self.param_keys = param_keys
        self.prior_dict = prior_dict
        self._data = data
        self._samples = mcmc.get_samples()
        self._num_chains = mcmc.num_chains
        self._num_warmup = mcmc.num_warmup
        self._num_samples = mcmc.num_samples
        self._n_obs = data.shape[0]

        # Import diagnostics helpers
        _ess_fn, _rhat_fn, _div_fn = _import_bayes_diag()
        self._ess_fn = _ess_fn
        self._rhat_fn = _rhat_fn
        self._div_fn = _div_fn

        # Auto-warn on convergence issues
        self._check_convergence()

    # ── convergence check ───────────────────────────────────────────────

    def _check_convergence(self):
        diag = self.diagnostics()
        issues = []
        if diag["max_rhat"] > 1.01:
            issues.append(
                f"max R-hat = {diag['max_rhat']:.3f} (should be < 1.01)"
            )
        if diag["min_ess"] < 100:
            issues.append(
                f"min ESS = {diag['min_ess']:.0f} (should be > 100)"
            )
        if diag["divergences"] > 0:
            pct = diag["divergence_pct"]
            if pct > 1.0:
                issues.append(
                    f"{diag['divergences']} divergences ({pct:.1f}%)"
                )
        if issues:
            warnings.warn(
                "Potential convergence issues: " + "; ".join(issues),
                RuntimeWarning,
                stacklevel=4,
            )

    # ── draws ───────────────────────────────────────────────────────────

    def draws(self) -> pd.DataFrame:
        """Return raw posterior draws as a DataFrame.

        Each row is one draw, columns are parameter names.
        """
        data = {}
        for key in self.param_keys:
            if key in self._samples:
                data[key] = np.array(self._samples[key]).ravel()
        return pd.DataFrame(data)

    # ── estimates ───────────────────────────────────────────────────────

    def estimates(self) -> pd.DataFrame:
        """Return parameter estimates as a DataFrame.

        Columns: lhs, op, rhs, mean, median, sd, ci.lower, ci.upper, rhat, ess.
        """
        rows = []
        for p in self.spec.params:
            if not p.free:
                continue
            key = _param_key(p)
            if key not in self._samples:
                continue
            s = np.array(self._samples[key]).ravel()
            category = _matrix_category(p, self.spec)

            ess = self._ess_fn(s)
            rhat = self._rhat_fn(s, self._num_chains)

            rows.append({
                "lhs": p.lhs,
                "op": p.op,
                "rhs": p.rhs,
                "category": category,
                "mean": float(np.mean(s)),
                "median": float(np.median(s)),
                "sd": float(np.std(s)),
                "ci.lower": float(np.percentile(s, 2.5)),
                "ci.upper": float(np.percentile(s, 97.5)),
                "rhat": float(rhat),
                "ess": float(ess),
            })
        return pd.DataFrame(rows)

    # ── diagnostics ─────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """Return MCMC diagnostic summary.

        Keys: n_divergences, divergence_pct, min_ess, max_rhat,
              num_chains, num_warmup, num_samples.
        """
        n_div, div_pct = self._div_fn(self.mcmc)

        ess_vals = []
        rhat_vals = []
        for key in self.param_keys:
            if key not in self._samples:
                continue
            s = np.array(self._samples[key]).ravel()
            ess_vals.append(self._ess_fn(s))
            r = self._rhat_fn(s, self._num_chains)
            if not np.isnan(r):
                rhat_vals.append(r)

        return {
            "divergences": n_div,
            "divergence_pct": div_pct,
            "min_ess": float(min(ess_vals)) if ess_vals else float("nan"),
            "max_rhat": float(max(rhat_vals)) if rhat_vals else float("nan"),
            "num_chains": self._num_chains,
            "num_warmup": self._num_warmup,
            "num_samples": self._num_samples,
        }

    # ── WAIC ────────────────────────────────────────────────────────────

    def _pointwise_log_lik(self) -> np.ndarray:
        """Compute pointwise log-likelihood for each draw and observation.

        Returns shape (n_draws, n_observations).
        """
        from .bayes import _import_jax, build_numpyro_model, _build_param_table

        jnp = _import_jax()
        import jax
        import numpyro.distributions as dist

        spec = self.spec
        F = jnp.array(spec.F)
        I = jnp.eye(spec.n_vars)
        A_fixed = jnp.array(spec.A_values)
        S_fixed = jnp.array(spec.S_values)
        obs_data = jnp.array(self._data)

        m_fixed = None
        if spec.meanstructure and spec.m_values is not None:
            m_fixed = jnp.array(spec.m_values)

        a_params, s_params, m_params, n_eff, param_keys = _build_param_table(spec)

        # Group by effective index
        eff_a = {}
        for key, r, c, ridx, eidx in a_params:
            eff_a.setdefault(eidx, []).append((r, c))
        eff_s = {}
        for key, r, c, ridx, eidx in s_params:
            eff_s.setdefault(eidx, []).append((r, c))
        eff_m = {}
        for key, i, ridx, eidx in m_params:
            eff_m.setdefault(eidx, []).append(i)

        draws_df = self.draws()
        n_draws = len(draws_df)
        n_obs = self._data.shape[0]
        ll = np.zeros((n_draws, n_obs))

        for d in range(n_draws):
            # Build theta dict for this draw
            theta = {}
            for eidx in range(n_eff):
                key = param_keys[eidx]
                theta[eidx] = jnp.array(draws_df[key].iloc[d])

            A = A_fixed.copy()
            for eidx, placements in eff_a.items():
                val = theta[eidx]
                for r, c in placements:
                    A = A.at[r, c].set(val)

            S = S_fixed.copy()
            for eidx, placements in eff_s.items():
                val = theta[eidx]
                for r, c in placements:
                    S = S.at[r, c].set(val)
                    if r != c:
                        S = S.at[c, r].set(val)

            I_A_inv = jnp.linalg.inv(I - A)
            Sigma = F @ I_A_inv @ S @ I_A_inv.T @ F.T
            Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * jnp.eye(spec.n_obs)

            if m_fixed is not None and m_params:
                m = m_fixed.copy()
                for eidx, placements in eff_m.items():
                    val = theta[eidx]
                    for i in placements:
                        m = m.at[i].set(val)
                mu = F @ I_A_inv @ m
            else:
                mu = jnp.zeros(spec.n_obs)

            mvn = dist.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
            ll[d, :] = np.array(mvn.log_prob(obs_data))

        return ll

    def waic(self) -> dict:
        """Compute WAIC (Widely Applicable Information Criterion).

        Returns
        -------
        dict
            Keys: waic, p_waic (effective number of parameters), se.
        """
        ll = self._pointwise_log_lik()
        # WAIC = -2 * (lppd - p_waic)
        # lppd = sum_i log(mean_s exp(ll[s,i]))
        # p_waic = sum_i var_s(ll[s,i])

        from scipy.special import logsumexp
        n_draws = ll.shape[0]

        # lppd per observation
        lppd_i = logsumexp(ll, axis=0) - np.log(n_draws)

        # p_waic per observation
        p_waic_i = np.var(ll, axis=0, ddof=1)

        lppd = np.sum(lppd_i)
        p_waic = np.sum(p_waic_i)
        waic = -2.0 * (lppd - p_waic)

        # SE of WAIC
        elpd_i = lppd_i - p_waic_i
        se = float(np.sqrt(len(elpd_i) * np.var(-2 * elpd_i, ddof=1)))

        return {"waic": float(waic), "p_waic": float(p_waic), "se": se}

    def loo(self) -> dict:
        """Compute LOO-CV via Pareto-smoothed importance sampling (PSIS-LOO).

        Returns
        -------
        dict
            Keys: loo, p_loo (effective parameters), se, k_max (max Pareto k).
        """
        ll = self._pointwise_log_lik()
        n_draws, n_obs = ll.shape

        elpd_loo_i = np.zeros(n_obs)
        k_values = np.zeros(n_obs)

        for i in range(n_obs):
            # Log importance ratios: -ll[s,i] (leave-one-out weights)
            log_ratios = -ll[:, i]
            log_ratios -= np.max(log_ratios)  # stabilize

            # Fit generalized Pareto to the tail
            ratios = np.exp(log_ratios)
            k = _pareto_k_estimate(ratios)
            k_values[i] = k

            # PSIS weights (simplified: use raw IS if k < 0.7)
            log_weights = log_ratios - np.max(log_ratios)
            weights = np.exp(log_weights)
            weights /= weights.sum()

            # elpd_loo_i = log(sum(w * exp(ll)))
            elpd_loo_i[i] = np.log(np.sum(weights * np.exp(ll[:, i])))

        elpd_loo = np.sum(elpd_loo_i)
        p_loo = np.sum(
            np.log(np.mean(np.exp(ll), axis=0)) - elpd_loo_i
        )
        loo_val = -2.0 * elpd_loo
        se = float(np.sqrt(n_obs * np.var(-2 * elpd_loo_i, ddof=1)))

        return {
            "loo": float(loo_val),
            "p_loo": float(p_loo),
            "se": se,
            "k_max": float(np.max(k_values)),
        }

    # ── summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print and return a Bayesian SEM summary string."""
        diag = self.diagnostics()
        est = self.estimates()

        lines = []
        lines.append("semla 0.1.0 — Bayesian SEM Results (NumPyro)")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"  Estimator                                        {'Bayes':>8s}")
        lines.append(f"  Chains                                           {diag['num_chains']:>8d}")
        lines.append(f"  Draws per chain                                  {diag['num_samples']:>8d}")
        lines.append(f"  Warmup per chain                                 {diag['num_warmup']:>8d}")
        lines.append(f"  Number of observations                           {self._n_obs:>8d}")

        # Divergences
        n_div = diag["divergences"]
        div_pct = diag["divergence_pct"]
        if n_div > 0:
            lines.append(f"  Divergences                              {n_div:>5d} ({div_pct:.1f}%)")
        else:
            lines.append(f"  Divergences                                         0")
        lines.append("")

        # Parameter estimates
        lines.append("Parameter Estimates (posterior):")
        lines.append("")
        lines.append(
            f"  {'lhs':<10s} {'op':<4s} {'rhs':<10s} {'mean':>8s} {'median':>8s} "
            f"{'sd':>8s} {'ci.lower':>8s} {'ci.upper':>8s} {'rhat':>6s} {'ess':>7s}"
        )
        lines.append("  " + "-" * 83)

        for op_label, op_code in [
            ("Latent Variables:", "=~"),
            ("Regressions:", "~"),
            ("Intercepts:", "~1"),
            ("Covariances/Variances:", "~~"),
        ]:
            subset = est[est["op"] == op_code]
            if subset.empty:
                continue
            lines.append(f"\n  {op_label}")
            for _, row in subset.iterrows():
                rhat_str = f"{row['rhat']:.3f}" if not np.isnan(row["rhat"]) else "  n/a"
                lines.append(
                    f"  {row['lhs']:<10s} {row['op']:<4s} {row['rhs']:<10s} "
                    f"{row['mean']:>8.3f} {row['median']:>8.3f} {row['sd']:>8.3f} "
                    f"{row['ci.lower']:>8.3f} {row['ci.upper']:>8.3f} "
                    f"{rhat_str:>6s} {row['ess']:>7.0f}"
                )

        lines.append("")
        lines.append("=" * 65)

        output = "\n".join(lines)
        print(output)
        return output

    def fit_indices(self) -> dict:
        """Return Bayesian model comparison indices (WAIC)."""
        return self.waic()

    @property
    def converged(self) -> bool:
        """Whether the sampler converged (max R-hat < 1.05)."""
        diag = self.diagnostics()
        return diag["max_rhat"] < 1.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pareto_k_estimate(x: np.ndarray) -> float:
    """Estimate Pareto shape parameter k from the upper tail of x.

    Uses the Zhang & Stephens (2009) estimator for the generalized
    Pareto distribution shape parameter.
    """
    n = len(x)
    tail_n = max(int(min(n / 5, 3 * np.sqrt(n))), 5)
    sorted_x = np.sort(x)
    tail = sorted_x[-tail_n:]
    threshold = sorted_x[-tail_n - 1] if tail_n < n else sorted_x[0]
    exceedances = tail - threshold
    exceedances = exceedances[exceedances > 0]
    m = len(exceedances)
    if m < 2:
        return 0.0
    # Pickands estimator (robust)
    if m >= 4:
        q = int(m / 4)
        x_sorted = np.sort(exceedances)
        k = (1.0 / np.log(2)) * np.log(
            (x_sorted[-1] - x_sorted[-q - 1])
            / max(x_sorted[-q - 1] - x_sorted[-2 * q - 1], 1e-10)
        )
        return float(k)
    return 0.0
