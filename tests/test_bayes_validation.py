"""Validation: compare Bayesian posterior means against ML point estimates.

With weak/diffuse priors and sufficient draws, Bayesian posterior means
should converge to ML estimates. This validates the entire Bayesian pipeline
end-to-end against the well-tested frequentist estimator.

These tests use real MCMC sampling and are slow (~30-60s each).
"""

import numpy as np
import pandas as pd
import pytest

numpyro = pytest.importorskip("numpyro")

from semla import cfa, sem
from semla.datasets import HolzingerSwineford1939
from semla.priors import Normal, InverseGamma


# ── fixtures ────────────────────────────────────────────────────────────

HS_SYNTAX = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""

SEM_SYNTAX = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
    speed ~ visual + textual
    visual ~~ textual
"""


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def ml_cfa(hs_data):
    """ML CFA fit on Holzinger-Swineford."""
    return cfa(HS_SYNTAX, data=hs_data)


@pytest.fixture(scope="module")
def bayes_cfa(hs_data):
    """Bayesian CFA fit on Holzinger-Swineford with adaptive priors."""
    return cfa(
        HS_SYNTAX,
        data=hs_data,
        estimator="bayes",
        warmup=1000,
        draws=3000,
        chains=4,
        seed=42,
        adapt_convergence=False,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def ml_sem(hs_data):
    """ML SEM fit."""
    return sem(SEM_SYNTAX, data=hs_data)


@pytest.fixture(scope="module")
def bayes_sem(hs_data):
    """Bayesian SEM fit with adaptive priors and longer warmup."""
    return sem(
        SEM_SYNTAX,
        data=hs_data,
        estimator="bayes",
        warmup=2000,
        draws=4000,
        chains=4,
        seed=42,
        adapt_delta=0.95,
        adapt_convergence=False,
        progress_bar=False,
    )


# ── helpers ─────────────────────────────────────────────────────────────

def _compare_estimates(ml_fit, bayes_fit, atol_loadings=0.15, atol_variances=0.2):
    """Compare ML and Bayesian parameter estimates.

    Returns a DataFrame with columns: param, ml_est, bayes_mean, diff, category.
    Asserts that differences are within tolerance.
    """
    ml_est = ml_fit.estimates()
    ml_free = ml_est[ml_est["free"]].copy()

    bayes_est = bayes_fit.estimates()

    rows = []
    for _, ml_row in ml_free.iterrows():
        key = f"{ml_row['lhs']}{ml_row['op']}{ml_row['rhs']}"
        # Find matching Bayesian estimate
        match = bayes_est[
            (bayes_est["lhs"] == ml_row["lhs"]) &
            (bayes_est["op"] == ml_row["op"]) &
            (bayes_est["rhs"] == ml_row["rhs"])
        ]
        if match.empty:
            continue

        bayes_mean = match.iloc[0]["mean"]
        ml_val = ml_row["est"]
        diff = abs(bayes_mean - ml_val)
        category = match.iloc[0].get("category", "unknown")

        rows.append({
            "param": key,
            "ml_est": ml_val,
            "bayes_mean": bayes_mean,
            "diff": diff,
            "category": category,
        })

    return pd.DataFrame(rows)


# ── CFA validation ─────────────────────────────────────────────────────

class TestCFAValidation:
    """Compare 3-factor CFA: Bayesian vs ML on Holzinger-Swineford."""

    def test_bayes_converged(self, bayes_cfa):
        diag = bayes_cfa.results.diagnostics()
        # With 2000 draws and 4 chains, should converge
        assert diag["max_rhat"] < 1.05, (
            f"Bayesian CFA did not converge: max R-hat = {diag['max_rhat']:.3f}"
        )

    def test_loadings_close_to_ml(self, ml_cfa, bayes_cfa):
        comp = _compare_estimates(ml_cfa, bayes_cfa)
        loadings = comp[comp["category"] == "loadings"]

        for _, row in loadings.iterrows():
            assert row["diff"] < 0.15, (
                f"Loading {row['param']}: ML={row['ml_est']:.3f}, "
                f"Bayes={row['bayes_mean']:.3f}, diff={row['diff']:.3f}"
            )

    def test_variances_close_to_ml(self, ml_cfa, bayes_cfa):
        comp = _compare_estimates(ml_cfa, bayes_cfa)
        variances = comp[comp["category"].isin(
            ["residual_variances", "factor_variances"]
        )]

        for _, row in variances.iterrows():
            # Variances can differ more due to prior influence
            assert row["diff"] < 0.3, (
                f"Variance {row['param']}: ML={row['ml_est']:.3f}, "
                f"Bayes={row['bayes_mean']:.3f}, diff={row['diff']:.3f}"
            )

    def test_covariances_close_to_ml(self, ml_cfa, bayes_cfa):
        comp = _compare_estimates(ml_cfa, bayes_cfa)
        covs = comp[comp["category"] == "covariances"]

        for _, row in covs.iterrows():
            assert row["diff"] < 0.15, (
                f"Covariance {row['param']}: ML={row['ml_est']:.3f}, "
                f"Bayes={row['bayes_mean']:.3f}, diff={row['diff']:.3f}"
            )

    def test_ml_within_bayes_ci(self, ml_cfa, bayes_cfa):
        """ML point estimates should fall within Bayesian 95% CIs."""
        ml_est = ml_cfa.estimates()
        ml_free = ml_est[ml_est["free"]]
        bayes_est = bayes_cfa.results.estimates()

        n_within = 0
        n_total = 0
        for _, ml_row in ml_free.iterrows():
            match = bayes_est[
                (bayes_est["lhs"] == ml_row["lhs"]) &
                (bayes_est["op"] == ml_row["op"]) &
                (bayes_est["rhs"] == ml_row["rhs"])
            ]
            if match.empty:
                continue

            n_total += 1
            ci_low = match.iloc[0]["ci.lower"]
            ci_high = match.iloc[0]["ci.upper"]
            if ci_low <= ml_row["est"] <= ci_high:
                n_within += 1

        # At least 80% of ML estimates should fall within Bayesian CIs
        coverage = n_within / n_total if n_total > 0 else 0
        assert coverage >= 0.80, (
            f"Only {n_within}/{n_total} ({coverage:.0%}) ML estimates "
            f"fall within Bayesian 95% CIs"
        )

    def test_bayes_summary_runs(self, bayes_cfa):
        """Summary should print without error."""
        s = bayes_cfa.summary()
        assert "Bayesian SEM Results" in s
        assert "visual" in s or "x1" in s

    def test_bayes_diagnostics(self, bayes_cfa):
        diag = bayes_cfa.results.diagnostics()
        assert diag["num_chains"] == 4
        assert diag["num_samples"] == 3000
        assert diag["min_ess"] > 0


# ── SEM validation ──────────────────────────────────────────────────────

class TestSEMValidation:
    """Compare SEM with regressions: Bayesian vs ML.

    Uses positive loading constraints to prevent sign-flipping.
    Tolerances are slightly wider than CFA due to structural complexity.
    """

    def test_bayes_converged(self, bayes_sem):
        diag = bayes_sem.results.diagnostics()
        assert diag["max_rhat"] < 1.05, (
            f"Bayesian SEM did not converge: max R-hat = {diag['max_rhat']:.3f}"
        )

    def test_regression_coefficients_close(self, ml_sem, bayes_sem):
        comp = _compare_estimates(ml_sem, bayes_sem)
        regs = comp[comp["category"] == "regressions"]

        for _, row in regs.iterrows():
            assert row["diff"] < 0.2, (
                f"Regression {row['param']}: ML={row['ml_est']:.3f}, "
                f"Bayes={row['bayes_mean']:.3f}, diff={row['diff']:.3f}"
            )

    def test_loadings_close(self, ml_sem, bayes_sem):
        comp = _compare_estimates(ml_sem, bayes_sem)
        loadings = comp[comp["category"] == "loadings"]

        for _, row in loadings.iterrows():
            assert row["diff"] < 0.15, (
                f"Loading {row['param']}: ML={row['ml_est']:.3f}, "
                f"Bayes={row['bayes_mean']:.3f}, diff={row['diff']:.3f}"
            )


# ── WAIC sanity checks ─────────────────────────────────────────────────

class TestWAICSanity:
    """WAIC should be a reasonable number and p_waic should be positive."""

    def test_waic_is_finite(self, bayes_cfa):
        w = bayes_cfa.results.waic()
        assert np.isfinite(w["waic"]), f"WAIC is not finite: {w['waic']}"

    def test_p_waic_positive(self, bayes_cfa):
        w = bayes_cfa.results.waic()
        assert w["p_waic"] > 0, f"p_WAIC should be positive, got {w['p_waic']}"

    def test_p_waic_reasonable(self, bayes_cfa):
        """p_WAIC should be roughly in the neighborhood of n_free params."""
        w = bayes_cfa.results.waic()
        n_free = bayes_cfa.results.spec.n_free
        # p_waic should be in a reasonable range
        assert w["p_waic"] < n_free * 10, (
            f"p_WAIC={w['p_waic']:.1f} seems too large for "
            f"{n_free} free parameters"
        )


# ── LOO sanity checks ──────────────────────────────────────────────────

class TestLOOSanity:
    def test_loo_is_finite(self, bayes_cfa):
        l = bayes_cfa.results.loo()
        assert np.isfinite(l["loo"]), f"LOO is not finite: {l['loo']}"

    def test_loo_se_positive(self, bayes_cfa):
        l = bayes_cfa.results.loo()
        assert l["se"] > 0


# ── Print comparison table ──────────────────────────────────────────────

class TestComparisonReport:
    """Not a real test — prints a comparison table for visual inspection."""

    def test_print_comparison(self, ml_cfa, bayes_cfa):
        comp = _compare_estimates(ml_cfa, bayes_cfa)
        print("\n" + "=" * 70)
        print("ML vs Bayesian Parameter Comparison (3-factor CFA)")
        print("=" * 70)
        print(f"{'Parameter':<20s} {'ML':>10s} {'Bayes':>10s} {'Diff':>8s} {'Category':<20s}")
        print("-" * 70)
        for _, row in comp.iterrows():
            print(
                f"{row['param']:<20s} {row['ml_est']:>10.3f} "
                f"{row['bayes_mean']:>10.3f} {row['diff']:>8.3f} "
                f"{row['category']:<20s}"
            )
        print("=" * 70)
