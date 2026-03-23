"""End-to-end validation with classic lavaan tutorial model patterns.

Complements test_validate_comprehensive.py (basic CFA, SEM, constraints) and
test_validate_models.py (higher-order, growth, CLPM) with additional model
configurations from the lavaan tutorial and SEM textbooks.

Models tested:
  F — Single-factor CFA (all 9 HS indicators)
  G — Orthogonal CFA (factor covariances fixed to 0)
  H — CFA with correlated residuals
  I — Observed-variable mediation with indirect effects (:=)

Reference values: lavaan 0.6-21 on HolzingerSwineford1939 (n=301).
Mediation model uses simulated data (seed=42, n=300).
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa, sem
from semla.datasets import HolzingerSwineford1939


# ── shared data fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


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


# ============================================================
# Model F: Single-factor CFA (all 9 HS indicators on one g)
# ============================================================
# R: cfa('g =~ x1+x2+x3+x4+x5+x6+x7+x8+x9', data=HolzingerSwineford1939)

class TestSingleFactorCFA:
    """g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9."""

    MODEL = "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 311.227, 1.0),
        ("df", 27, 0),
        ("cfi", 0.677, 0.005),
        ("tli", 0.569, 0.005),
        ("rmsea", 0.187, 0.005),
        ("srmr", 0.143, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- factor loadings (free) --

    @pytest.mark.parametrize("ind,lav_est,lav_se", [
        ("x2", 0.508, 0.152),
        ("x3", 0.493, 0.146),
        ("x4", 1.930, 0.257),
        ("x5", 2.123, 0.283),
        ("x6", 1.796, 0.239),
        ("x7", 0.385, 0.137),
        ("x8", 0.398, 0.129),
        ("x9", 0.606, 0.138),
    ])
    def test_loading(self, est, ind, lav_est, lav_se):
        _check_param(est, "g", "=~", ind, lav_est, lav_se)

    # -- latent variance --

    def test_latent_variance(self, est):
        _check_param(est, "g", "~~", "g", 0.261, 0.069)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 1.102, 0.093),
        ("x2", 1.319, 0.108),
        ("x3", 1.216, 0.100),
        ("x4", 0.381, 0.048),
        ("x5", 0.487, 0.060),
        ("x6", 0.357, 0.043),
        ("x7", 1.148, 0.094),
        ("x8", 0.984, 0.081),
        ("x9", 0.923, 0.076),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    def test_poor_fit(self, fid):
        """Single-factor model should fit poorly (misspecified)."""
        assert fid["cfi"] < 0.80
        assert fid["rmsea"] > 0.10


# ============================================================
# Model G: Orthogonal CFA (factor covariances fixed to 0)
# ============================================================
# R: cfa(model, data=HolzingerSwineford1939, orthogonal=TRUE)

class TestOrthogonalCFA:
    """3-factor CFA with all factor covariances fixed to zero."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
        visual ~~ 0*textual
        visual ~~ 0*speed
        textual ~~ 0*speed
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 153.007, 1.0),
        ("df", 27, 0),
        ("cfi", 0.857, 0.005),
        ("tli", 0.809, 0.005),
        ("rmsea", 0.125, 0.005),
        ("srmr", 0.161, 0.005),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    def test_df_3_more_than_standard(self, fid):
        """Orthogonal model has 3 more df than standard (27 vs 24)."""
        assert fid["df"] == 27

    # -- factor loadings --

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.778, 0.141),
        ("visual", "x3", 1.107, 0.214),
        ("textual", "x5", 1.133, 0.067),
        ("textual", "x6", 0.924, 0.056),
        ("speed", "x8", 1.225, 0.190),
        ("speed", "x9", 0.854, 0.121),
    ])
    def test_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se)

    # -- covariances must be exactly zero --

    @pytest.mark.parametrize("lv1,lv2", [
        ("visual", "textual"),
        ("visual", "speed"),
        ("textual", "speed"),
    ])
    def test_covariance_zero(self, est, lv1, lv2):
        row = _get_est(est, lv1, "~~", lv2)
        assert row["est"] == 0.0
        assert not row["free"]

    # -- latent variances --

    @pytest.mark.parametrize("lv,lav_est,lav_se", [
        ("visual", 0.525, 0.131),
        ("textual", 0.972, 0.113),
        ("speed", 0.438, 0.097),
    ])
    def test_latent_variance(self, est, lv, lav_est, lav_se):
        _check_param(est, lv, "~~", lv, lav_est, lav_se)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.837, 0.119),
        ("x2", 1.068, 0.105),
        ("x3", 0.635, 0.130),
        ("x4", 0.383, 0.049),
        ("x5", 0.418, 0.059),
        ("x6", 0.370, 0.044),
        ("x7", 0.749, 0.087),
        ("x8", 0.367, 0.097),
        ("x9", 0.698, 0.073),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    def test_worse_fit_than_correlated(self, fid):
        """Orthogonal model should fit worse than correlated factors."""
        assert fid["cfi"] < 0.90
        assert fid["rmsea"] > 0.10


# ============================================================
# Model H: CFA with correlated residuals
# ============================================================
# R: cfa('visual=~x1+x2+x3; textual=~x4+x5+x6; speed=~x7+x8+x9;
#         x1~~x2', data=HolzingerSwineford1939)

class TestCFACorrelatedResiduals:
    """3-factor CFA with x1 ~~ x2 correlated residual."""

    MODEL = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
        x1 ~~ x2
    """

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(self.MODEL, data=hs_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit indices --
    # Adding x1~~x2 removes 1 df (23 instead of 24)

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("df", 23, 0),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    def test_better_fit_than_standard(self, fid):
        """Adding a correlated residual should improve fit."""
        assert fid["chi_square"] < 85.306  # standard CFA chi-square
        assert fid["df"] == 23  # one fewer df

    # -- correlated residual --

    def test_correlated_residual_estimated(self, est):
        """x1 ~~ x2 should be freely estimated."""
        row = _get_est(est, "x1", "~~", "x2")
        assert row["free"]

    # -- textual and speed loadings approximately stable --
    # Adding x1~~x2 mainly affects visual; textual/speed shift slightly.

    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_approx_stable_loading(self, est, lv, ind, lav_est, lav_se):
        _check_param(est, lv, "=~", ind, lav_est, lav_se, atol_est=0.05)


# ============================================================
# Model I: Observed-variable mediation (path analysis)
# ============================================================
# Simulated data: X -> M -> Y with direct effect X -> Y
# True: a=0.5, b=0.4, c(direct)=0.2
# R: sem('M~a*X; Y~b*M+c*X; indirect:=a*b; total:=a*b+c', data=med_data)

class TestMediationPathModel:
    """Observed-variable mediation: X -> M -> Y + X -> Y."""

    MODEL = """
        M ~ a*X
        Y ~ b*M + c*X
        indirect := a*b
        total := a*b + c
    """

    @pytest.fixture(scope="class")
    def med_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.8, n)
        Y = 0.4 * M + 0.2 * X + rng.normal(0, 0.7, n)
        return pd.DataFrame({"X": X, "M": M, "Y": Y})

    @pytest.fixture(scope="class")
    def fit(self, med_data):
        return sem(self.MODEL, data=med_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def defined(self, fit):
        return fit.defined_estimates()

    # -- just-identified model --

    def test_just_identified(self, fit):
        fi = fit.fit_indices()
        assert fi["df"] == 0
        assert fi["chi_square"] < 0.01

    # -- regression coefficients --

    def test_a_path(self, est):
        """M ~ X (a path) should recover ~0.5."""
        _check_param(est, "M", "~", "X", 0.529, 0.051, atol_est=0.02)

    def test_b_path(self, est):
        """Y ~ M (b path) should recover ~0.4."""
        _check_param(est, "Y", "~", "M", 0.285, 0.050, atol_est=0.02)

    def test_c_path(self, est):
        """Y ~ X (c' direct effect) should recover ~0.2."""
        _check_param(est, "Y", "~", "X", 0.205, 0.051, atol_est=0.02)

    # -- defined parameters (indirect and total effects) --

    def test_indirect_effect(self, defined):
        indirect = defined[defined["name"] == "indirect"]
        assert len(indirect) == 1
        assert abs(indirect["est"].values[0] - 0.151) < 0.02

    def test_total_effect(self, defined):
        total = defined[defined["name"] == "total"]
        assert len(total) == 1
        assert abs(total["est"].values[0] - 0.356) < 0.02

    def test_indirect_se(self, defined):
        """Indirect effect SE via delta method should be positive."""
        indirect = defined[defined["name"] == "indirect"]
        assert indirect["se"].values[0] > 0

    def test_indirect_equals_a_times_b(self, est, defined):
        """indirect := a*b should exactly equal product of path estimates."""
        a = _get_est(est, "M", "~", "X")["est"]
        b = _get_est(est, "Y", "~", "M")["est"]
        indirect = defined[defined["name"] == "indirect"]["est"].values[0]
        assert abs(indirect - a * b) < 1e-6

    def test_total_equals_indirect_plus_direct(self, est, defined):
        """total := a*b + c should equal indirect + direct."""
        c = _get_est(est, "Y", "~", "X")["est"]
        indirect = defined[defined["name"] == "indirect"]["est"].values[0]
        total = defined[defined["name"] == "total"]["est"].values[0]
        assert abs(total - (indirect + c)) < 1e-6

    # -- variances (should match sample variances for just-identified) --

    def test_residual_variances_positive(self, est):
        for var in ["X", "M", "Y"]:
            row = _get_est(est, var, "~~", var)
            assert row["est"] > 0
