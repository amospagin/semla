"""Validation of advanced model types against lavaan 0.6-20.

Model types tested:
- Higher-order (second-order) factor model
- Latent growth curve model

Reference values generated from lavaan 0.6-20 using R 4.5.3.
"""

import numpy as np
import pandas as pd
import pytest
from semla import sem, growth
from semla.datasets import HolzingerSwineford1939


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_est(estimates, lhs, op, rhs):
    """Extract a single parameter estimate row."""
    row = estimates[
        (estimates["lhs"] == lhs) &
        (estimates["op"] == op) &
        (estimates["rhs"] == rhs)
    ]
    assert len(row) == 1, f"Expected 1 row for {lhs} {op} {rhs}, got {len(row)}"
    return row.iloc[0]


def _check_param(estimates, lhs, op, rhs, lav_est, lav_se,
                 atol_est=0.01, atol_se=0.01):
    """Assert that estimate and SE match lavaan values."""
    row = _get_est(estimates, lhs, op, rhs)
    assert abs(row["est"] - lav_est) < atol_est, (
        f"{lhs} {op} {rhs}: est={row['est']:.6f}, lavaan={lav_est}, "
        f"diff={abs(row['est'] - lav_est):.6f}"
    )
    if lav_se > 0:
        assert abs(row["se"] - lav_se) < atol_se, (
            f"{lhs} {op} {rhs}: se={row['se']:.6f}, lavaan={lav_se}, "
            f"diff={abs(row['se'] - lav_se):.6f}"
        )


def _check_fit(fit_indices, measure, lavaan_val, atol):
    """Assert that a fit index matches lavaan."""
    semla_val = fit_indices[measure]
    assert abs(semla_val - lavaan_val) <= atol, (
        f"{measure}: semla={semla_val:.6f}, lavaan={lavaan_val}, "
        f"diff={abs(semla_val - lavaan_val):.6f}"
    )


# ── shared data fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def growth_data():
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "src" / "semla" / "datasets" / "growth_data.csv"
    return pd.read_csv(data_path)


# ============================================================
# Model A: Higher-order (second-order) factor model
# ============================================================
# R code:
#   model <- '
#     visual  =~ x1 + x2 + x3
#     textual =~ x4 + x5 + x6
#     speed   =~ x7 + x8 + x9
#     g =~ visual + textual + speed
#   '
#   fit <- cfa(model, data=HolzingerSwineford1939)

class TestHigherOrderCFA:
    """Second-order 'g' factor over visual, textual, speed."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
        g =~ visual + textual + speed
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return sem(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices (equivalent to correlated 3-factor CFA) --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 85.306, 0.5),
        ("df", 24, 0),
        ("cfi", 0.931, 0.005),
        ("tli", 0.896, 0.005),
        ("rmsea", 0.092, 0.005),
        ("srmr", 0.065, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- first-order factor loadings --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.554, 0.100),
        ("visual", "x3", 0.729, 0.109),
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_first_order_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- second-order factor loadings --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("textual", 0.662, 0.173),
        ("speed", 0.425, 0.118),
    ])
    def test_second_order_loading(self, est, lv, lav_est, lav_se):
        _check_param(est, "g", "=~", lv, lav_est, lav_se)

    # -- residual variances (observed) --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.549, 0.114),
        ("x2", 1.134, 0.102),
        ("x3", 0.844, 0.091),
        ("x4", 0.371, 0.048),
        ("x5", 0.446, 0.058),
        ("x6", 0.356, 0.043),
        ("x7", 0.799, 0.081),
        ("x8", 0.488, 0.074),
        ("x9", 0.566, 0.071),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- first-order factor residual variances --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("visual", 0.192, 0.170),
        ("textual", 0.709, 0.107),
        ("speed", 0.272, 0.069),
    ])
    def test_first_order_residual_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    # -- second-order factor variance --

    def test_g_variance(self, est):
        _check_param(est, "g", "~~", "g", 0.617, 0.183)


# ============================================================
# Model B: Linear latent growth curve model
# ============================================================
# R code:
#   set.seed(42)
#   n <- 200
#   intercept <- rnorm(n, mean=5, sd=1)
#   slope <- rnorm(n, mean=0.5, sd=0.5)
#   y1 <- intercept + 0*slope + rnorm(n, 0, 0.5)
#   y2 <- intercept + 1*slope + rnorm(n, 0, 0.5)
#   y3 <- intercept + 2*slope + rnorm(n, 0, 0.5)
#   y4 <- intercept + 3*slope + rnorm(n, 0, 0.5)
#   growth_data <- data.frame(y1, y2, y3, y4)
#
#   model <- '
#     i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
#     s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4
#   '
#   fit <- growth(model, data=growth_data)

class TestLinearGrowthCurve:
    """Linear growth model with 4 time points."""

    MODEL = """
        i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
        s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4
    """

    @pytest.fixture(scope="class")
    def fit(self, growth_data):
        return growth(self.MODEL, data=growth_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 3.571, 0.5),
        ("df", 5, 0),
        ("cfi", 1.000, 0.005),
        ("rmsea", 0.000, 0.02),
        ("srmr", 0.015, 0.01),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- latent means (intercepts) --

    def test_intercept_mean(self, est):
        """Mean of the intercept factor (~4.928)."""
        _check_param(est, "i", "~1", "1", 4.928, 0.073)

    def test_slope_mean(self, est):
        """Mean of the slope factor (~0.520)."""
        _check_param(est, "s", "~1", "1", 0.520, 0.035)

    # -- latent variances --

    def test_intercept_variance(self, est):
        _check_param(est, "i", "~~", "i", 0.870, 0.109)

    def test_slope_variance(self, est):
        _check_param(est, "s", "~~", "s", 0.189, 0.027)

    # -- intercept-slope covariance --

    def test_intercept_slope_covariance(self, est):
        _check_param(est, "i", "~~", "s", 0.003, 0.039, atol_est=0.02)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("y1", 0.292, 0.060),
        ("y2", 0.200, 0.032),
        ("y3", 0.301, 0.046),
        ("y4", 0.355, 0.087),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    # -- observed intercepts should be fixed to 0 --

    def test_observed_intercepts_fixed(self, est):
        """In a growth model, observed intercepts are fixed to 0."""
        intercepts = est[est["op"] == "~1"]
        obs_intercepts = intercepts[intercepts["lhs"].isin(["y1", "y2", "y3", "y4"])]
        # Should have no free observed intercepts
        assert len(obs_intercepts) == 0 or not obs_intercepts["free"].any(), (
            "Observed intercepts should be fixed to 0 in growth models"
        )

    def test_latent_means_estimated(self, est):
        """Intercept and slope means should be freely estimated."""
        intercepts = est[(est["op"] == "~1") & (est["free"])]
        latent_means = intercepts[intercepts["lhs"].isin(["i", "s"])]
        assert len(latent_means) == 2, (
            f"Expected 2 free latent means (i, s), got {len(latent_means)}"
        )
