"""Tests for AIC/BIC and R-squared."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_fit():
    df = HolzingerSwineford1939()
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=df)


class TestInformationCriteria:
    def test_aic_in_fit_indices(self, hs_fit):
        assert "aic" in hs_fit.fit_indices()

    def test_bic_in_fit_indices(self, hs_fit):
        assert "bic" in hs_fit.fit_indices()

    def test_abic_in_fit_indices(self, hs_fit):
        assert "abic" in hs_fit.fit_indices()

    def test_aic_greater_than_chi_square(self, hs_fit):
        idx = hs_fit.fit_indices()
        # AIC = chi_sq + 2k, so AIC > chi_sq
        assert idx["aic"] > idx["chi_square"]

    def test_bic_greater_than_aic(self, hs_fit):
        idx = hs_fit.fit_indices()
        # BIC > AIC when log(n) > 2, i.e. n > 7
        assert idx["bic"] > idx["aic"]

    def test_aic_formula(self, hs_fit):
        idx = hs_fit.fit_indices()
        k = hs_fit.results.n_free
        expected_aic = idx["chi_square"] + 2 * k
        assert abs(idx["aic"] - expected_aic) < 0.001

    def test_in_summary(self, hs_fit):
        summary = hs_fit.summary()
        assert "AIC" in summary
        assert "BIC" in summary


class TestRSquared:
    def test_returns_dict(self, hs_fit):
        r2 = hs_fit.r_squared()
        assert isinstance(r2, dict)

    def test_all_indicators_present(self, hs_fit):
        r2 = hs_fit.r_squared()
        for x in ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]:
            assert x in r2, f"{x} not in R-squared output"

    def test_values_between_0_and_1(self, hs_fit):
        r2 = hs_fit.r_squared()
        for var, val in r2.items():
            assert 0 <= val <= 1, f"R² for {var} = {val}, out of range"

    def test_textual_higher_than_visual(self, hs_fit):
        """Textual factor explains more variance (known from this dataset)."""
        r2 = hs_fit.r_squared()
        avg_textual = np.mean([r2["x4"], r2["x5"], r2["x6"]])
        avg_visual = np.mean([r2["x1"], r2["x2"], r2["x3"]])
        assert avg_textual > avg_visual

    def test_in_summary(self, hs_fit):
        summary = hs_fit.summary()
        assert "R-Square" in summary

    def test_latent_vars_not_in_r_squared(self, hs_fit):
        """Latent vars with no incoming regression should not appear."""
        r2 = hs_fit.r_squared()
        assert "visual" not in r2
        assert "textual" not in r2
        assert "speed" not in r2
