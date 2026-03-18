"""Integration tests using the public API."""

import numpy as np
import pandas as pd
import pytest
from semla import Model, cfa, sem


def _generate_three_factor_data(n=301, seed=42):
    """Generate data similar to Holzinger & Swineford (1939) structure."""
    rng = np.random.default_rng(seed)

    f1 = rng.normal(0, 1, n)  # visual
    f2 = rng.normal(0, 1, n)  # textual
    f3 = rng.normal(0, 1, n)  # speed

    # Add some factor correlations
    f2 = f2 + 0.3 * f1
    f3 = f3 + 0.2 * f1 + 0.2 * f2

    data = pd.DataFrame({
        "x1": 1.0 * f1 + rng.normal(0, 0.8, n),
        "x2": 0.6 * f1 + rng.normal(0, 0.9, n),
        "x3": 0.8 * f1 + rng.normal(0, 0.7, n),
        "x4": 1.0 * f2 + rng.normal(0, 0.5, n),
        "x5": 1.1 * f2 + rng.normal(0, 0.6, n),
        "x6": 0.9 * f2 + rng.normal(0, 0.5, n),
        "x7": 1.0 * f3 + rng.normal(0, 0.7, n),
        "x8": 0.7 * f3 + rng.normal(0, 0.8, n),
        "x9": 0.8 * f3 + rng.normal(0, 0.6, n),
    })
    return data


HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


class TestCFA:
    def test_cfa_converges(self):
        data = _generate_three_factor_data()
        fit = cfa(HS_MODEL, data)
        assert fit.converged

    def test_cfa_fit_indices(self):
        data = _generate_three_factor_data()
        fit = cfa(HS_MODEL, data)
        idx = fit.fit_indices()

        assert "chi_square" in idx
        assert "cfi" in idx
        assert "tli" in idx
        assert "rmsea" in idx
        assert "srmr" in idx

        # With well-structured data, fit should be reasonable
        assert idx["cfi"] > 0.85
        assert idx["rmsea"] < 0.15
        assert idx["srmr"] < 0.10

    def test_cfa_estimates_dataframe(self):
        data = _generate_three_factor_data()
        fit = cfa(HS_MODEL, data)
        est = fit.estimates()

        assert isinstance(est, pd.DataFrame)
        assert "est" in est.columns
        assert "se" in est.columns
        assert "z" in est.columns
        assert "pvalue" in est.columns

        # Should have loading, variance, and covariance parameters
        ops = est["op"].unique()
        assert "=~" in ops
        assert "~~" in ops

    def test_cfa_summary_runs(self):
        data = _generate_three_factor_data()
        fit = cfa(HS_MODEL, data)
        summary = fit.summary()
        assert "semla" in summary
        assert "CFI" in summary
        assert "RMSEA" in summary

    def test_cfa_loadings_positive(self):
        data = _generate_three_factor_data()
        fit = cfa(HS_MODEL, data)
        est = fit.estimates()

        loadings = est[est["op"] == "=~"]
        free_loadings = loadings[loadings["free"]]
        # All free loadings should be positive (data was generated with positive loadings)
        assert (free_loadings["est"] > 0).all()


class TestSEM:
    def test_sem_with_regression(self):
        data = _generate_three_factor_data()
        model = """
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            speed ~ visual + textual
        """
        fit = sem(model, data)
        assert fit.converged

        est = fit.estimates()
        regressions = est[est["op"] == "~"]
        assert len(regressions) == 2


class TestModel:
    def test_model_direct(self):
        data = _generate_three_factor_data()
        fit = Model(HS_MODEL, data)
        assert fit.converged

    def test_model_with_fixed_loading(self):
        data = _generate_three_factor_data()
        model = """
            visual  =~ 1*x1 + x2 + x3
            textual =~ 1*x4 + x5 + x6
        """
        fit = cfa(model, data)
        assert fit.converged

        est = fit.estimates()
        fixed = est[(est["op"] == "=~") & (~est["free"])]
        assert len(fixed) == 2
        assert (fixed["est"] == 1.0).all()
