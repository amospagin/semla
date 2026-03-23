"""Tests for large and complex models (#61).

Verifies semla handles models beyond the standard 9-indicator, 3-factor
test cases — including 5-factor, multi-layer mediation, and large samples.
"""

import time

import numpy as np
import pandas as pd
import pytest
from semla import cfa, sem


def _generate_factor_data(n_factors, n_indicators_per, n_obs, seed=42):
    """Generate data from a known multi-factor CFA model."""
    rng = np.random.default_rng(seed)
    n_total = n_factors * n_indicators_per

    # True loadings: 0.7 for all
    loading = 0.7
    # Factor correlations: 0.3 between all pairs
    factor_corr = 0.3

    # Factor covariance matrix
    Phi = np.full((n_factors, n_factors), factor_corr)
    np.fill_diagonal(Phi, 1.0)

    # Generate factor scores
    factors = rng.multivariate_normal(np.zeros(n_factors), Phi, size=n_obs)

    # Generate observed data
    data = {}
    for f in range(n_factors):
        for i in range(n_indicators_per):
            var_name = f"x{f * n_indicators_per + i + 1}"
            error = rng.normal(0, np.sqrt(1 - loading**2), n_obs)
            data[var_name] = loading * factors[:, f] + error

    return pd.DataFrame(data)


def _build_cfa_syntax(n_factors, n_indicators_per):
    """Build CFA syntax for a given number of factors and indicators."""
    lines = []
    for f in range(n_factors):
        indicators = [f"x{f * n_indicators_per + i + 1}"
                      for i in range(n_indicators_per)]
        lines.append(f"f{f+1} =~ {' + '.join(indicators)}")
    return "\n".join(lines)


class TestFiveFactorCFA:
    """5-factor CFA with 25 indicators."""

    @pytest.fixture(scope="class")
    def data(self):
        return _generate_factor_data(5, 5, 300)

    @pytest.fixture(scope="class")
    def fit(self, data):
        model = _build_cfa_syntax(5, 5)
        return cfa(model, data=data)

    def test_converges(self, fit):
        assert fit.converged

    def test_correct_df(self, fit):
        fi = fit.fit_indices()
        # 25*26/2 = 325 data points, 5*4 free loadings + 25 residuals +
        # 5 latent variances + 10 latent covariances = 60 free params
        assert fi["df"] == 265

    def test_reasonable_fit(self, fit):
        fi = fit.fit_indices()
        assert fi["cfi"] > 0.85
        assert fi["rmsea"] < 0.10

    def test_all_loadings_positive(self, fit):
        est = fit.estimates()
        loadings = est[est["op"] == "=~"]
        assert (loadings["est"] > 0).all()

    def test_check_runs(self, fit):
        result = fit.check()
        assert isinstance(result, str)


class TestLargeSample:
    """Standard CFA with n=5000 — tests speed and precision."""

    @pytest.fixture(scope="class")
    def fit(self):
        data = _generate_factor_data(3, 3, 5000)
        model = _build_cfa_syntax(3, 3)
        t0 = time.time()
        fit = cfa(model, data=data)
        elapsed = time.time() - t0
        fit._elapsed = elapsed
        return fit

    def test_converges(self, fit):
        assert fit.converged

    def test_fast(self, fit):
        assert fit._elapsed < 10.0, f"Took {fit._elapsed:.1f}s (expected <10s)"

    def test_tight_ses(self, fit):
        """Large sample should give small SEs."""
        est = fit.estimates()
        free_se = est[est["free"]]["se"]
        assert (free_se < 0.05).all(), f"Max SE: {free_se.max():.4f}"


class TestComplexSEM:
    """SEM with multiple mediation layers."""

    @pytest.fixture(scope="class")
    def fit(self):
        rng = np.random.default_rng(42)
        n = 500

        # Generate observed mediators and outcome
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        m1 = 0.4 * x1 + 0.3 * x2 + rng.normal(0, 0.8, n)
        m2 = 0.5 * m1 + 0.2 * x1 + rng.normal(0, 0.7, n)
        y = 0.3 * m2 + 0.2 * m1 + 0.1 * x1 + rng.normal(0, 0.6, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "m1": m1, "m2": m2, "y": y})

        return sem("""
            m1 ~ x1 + x2
            m2 ~ m1 + x1
            y  ~ m2 + m1 + x1
        """, data=df)

    def test_converges(self, fit):
        assert fit.converged

    def test_has_all_regressions(self, fit):
        est = fit.estimates()
        regs = est[est["op"] == "~"]
        # 2 + 2 + 3 = 7 regression paths
        assert len(regs) == 7

    def test_positive_r_squared(self, fit):
        r2 = fit.r_squared()
        for var, val in r2.items():
            assert val > 0, f"R² for {var} = {val}"


class TestTenFactorCFA:
    """10-factor CFA with 30 indicators — stress test."""

    @pytest.fixture(scope="class")
    def fit(self):
        data = _generate_factor_data(10, 3, 500)
        model = _build_cfa_syntax(10, 3)
        t0 = time.time()
        fit = cfa(model, data=data)
        elapsed = time.time() - t0
        fit._elapsed = elapsed
        return fit

    def test_converges(self, fit):
        assert fit.converged

    def test_under_30_seconds(self, fit):
        assert fit._elapsed < 30.0, f"Took {fit._elapsed:.1f}s"

    def test_reasonable_fit(self, fit):
        fi = fit.fit_indices()
        assert fi["cfi"] > 0.80
