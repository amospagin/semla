"""Tests for adaptive convergence (#30) and BayesianResults (#31)."""

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")

from semla.specification import build_specification
from semla.syntax import parse_syntax
from semla.bayes import (
    _max_rhat,
    _divergence_stats,
    _rhat,
    _effective_sample_size,
)
from semla.bayes_results import BayesianResults, _pareto_k_estimate


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_cfa_data():
    """Simple CFA with known parameters."""
    rng = np.random.default_rng(42)
    n = 200
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    x1 = 1.0 * f1 + rng.normal(0, 0.5, n)
    x2 = 0.8 * f1 + rng.normal(0, 0.5, n)
    x3 = 0.6 * f1 + rng.normal(0, 0.5, n)
    x4 = 1.0 * f2 + rng.normal(0, 0.5, n)
    x5 = 0.9 * f2 + rng.normal(0, 0.5, n)
    x6 = 0.7 * f2 + rng.normal(0, 0.5, n)
    data = np.column_stack([x1, x2, x3, x4, x5, x6])

    syntax = "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6"
    tokens = parse_syntax(syntax)
    spec = build_specification(tokens, ["x1", "x2", "x3", "x4", "x5", "x6"])
    return spec, data


# ── adaptive convergence helpers ────────────────────────────────────────

class TestMaxRhat:
    def test_converged_samples(self):
        rng = np.random.default_rng(42)
        samples = {
            "a": rng.normal(0, 1, 4000),
            "b": rng.normal(0, 1, 4000),
        }
        r = _max_rhat(samples, 4)
        assert r < 1.05

    def test_skips_obs_key(self):
        rng = np.random.default_rng(42)
        samples = {
            "a": rng.normal(0, 1, 4000),
            "obs": rng.normal(0, 1, (4000, 6)),  # observation likelihood
        }
        r = _max_rhat(samples, 4)
        assert r < 1.05


class TestDivergenceStats:
    def test_no_divergences_returns_zero(self):
        """When mcmc has no diverging field, return 0."""
        class FakeMCMC:
            def get_extra_fields(self):
                return {}
        n, pct = _divergence_stats(FakeMCMC())
        assert n == 0
        assert pct == 0.0

    def test_with_divergences(self):
        class FakeMCMC:
            def get_extra_fields(self):
                div = np.zeros(1000, dtype=bool)
                div[:30] = True  # 3% divergent
                return {"diverging": div}
        n, pct = _divergence_stats(FakeMCMC())
        assert n == 30
        assert abs(pct - 3.0) < 0.1


# ── BayesianResults diagnostics ─────────────────────────────────────────

class TestParetoK:
    def test_light_tail(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1, 1000)
        k = _pareto_k_estimate(x)
        # Should return a finite value for light-tailed data
        assert np.isfinite(k)

    def test_constant_returns_zero(self):
        x = np.ones(100)
        k = _pareto_k_estimate(x)
        assert k == 0.0


# ── MCMC integration (short run) ───────────────────────────────────────

class TestBayesianResultsIntegration:
    @pytest.fixture(scope="class")
    def bayes_result(self, simple_cfa_data):
        from semla.bayes import run_mcmc
        spec, data = simple_cfa_data
        return run_mcmc(
            spec, data,
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            seed=0,
            adapt_convergence=False,
            progress_bar=False,
        )

    def test_is_bayesian_results(self, bayes_result):
        assert isinstance(bayes_result, BayesianResults)

    def test_draws_shape(self, bayes_result, simple_cfa_data):
        spec, _ = simple_cfa_data
        df = bayes_result.draws()
        assert df.shape[0] == 100  # 2 chains * 50 draws
        assert df.shape[1] == spec.n_free

    def test_estimates_columns(self, bayes_result):
        est = bayes_result.estimates()
        for col in ["lhs", "op", "rhs", "mean", "median", "sd",
                     "ci.lower", "ci.upper", "rhat", "ess"]:
            assert col in est.columns

    def test_diagnostics_keys(self, bayes_result):
        diag = bayes_result.diagnostics()
        for key in ["divergences", "divergence_pct", "min_ess",
                     "max_rhat", "num_chains", "num_warmup", "num_samples"]:
            assert key in diag

    def test_summary_returns_string(self, bayes_result):
        s = bayes_result.summary()
        assert isinstance(s, str)
        assert "Bayesian SEM Results" in s
        assert "Bayes" in s

    def test_converged_property(self, bayes_result):
        # With 50 draws we may or may not converge, just check it doesn't crash
        assert isinstance(bayes_result.converged, bool)

    def test_fit_indices_returns_waic(self, bayes_result):
        fi = bayes_result.fit_indices()
        assert "waic" in fi
        assert "p_waic" in fi


class TestAdaptiveRun:
    """Test that adaptive convergence doesn't crash (short run)."""

    def test_adaptive_runs(self, simple_cfa_data):
        from semla.bayes import run_mcmc
        spec, data = simple_cfa_data
        result = run_mcmc(
            spec, data,
            num_warmup=30,
            num_samples=30,
            num_chains=2,
            seed=42,
            adapt_convergence=True,
            max_retries=1,
            progress_bar=False,
        )
        assert isinstance(result, BayesianResults)
