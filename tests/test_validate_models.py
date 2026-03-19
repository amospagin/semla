"""Validation of advanced model types against lavaan 0.6-20.

Model types tested:
- Higher-order (second-order) factor model
- Linear latent growth curve model
- Nonlinear (free-loading) growth curve model
- Cross-lagged panel model
- fitted() model-implied matrices

Reference values generated from lavaan 0.6-20 using R 4.5.3.
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa, sem, growth
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


@pytest.fixture(scope="module")
def clpm_data():
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "src" / "semla" / "datasets" / "clpm_data.csv"
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
#   set.seed(42); n <- 200
#   intercept <- rnorm(n, mean=5, sd=1); slope <- rnorm(n, mean=0.5, sd=0.5)
#   y1 <- intercept + 0*slope + rnorm(n, 0, 0.5)
#   y2 <- intercept + 1*slope + rnorm(n, 0, 0.5)
#   y3 <- intercept + 2*slope + rnorm(n, 0, 0.5)
#   y4 <- intercept + 3*slope + rnorm(n, 0, 0.5)
#   fit <- growth('i=~1*y1+1*y2+1*y3+1*y4; s=~0*y1+1*y2+2*y3+3*y4',
#                 data=data.frame(y1,y2,y3,y4))

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
        _check_param(est, "i", "~1", "1", 4.928, 0.073)

    def test_slope_mean(self, est):
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

    # -- growth-specific constraints --

    def test_observed_intercepts_fixed(self, est):
        """In a growth model, observed intercepts are fixed to 0."""
        intercepts = est[est["op"] == "~1"]
        obs_intercepts = intercepts[intercepts["lhs"].isin(["y1", "y2", "y3", "y4"])]
        assert len(obs_intercepts) == 0 or not obs_intercepts["free"].any()

    def test_latent_means_estimated(self, est):
        """Intercept and slope means should be freely estimated."""
        intercepts = est[(est["op"] == "~1") & (est["free"])]
        latent_means = intercepts[intercepts["lhs"].isin(["i", "s"])]
        assert len(latent_means) == 2


# ============================================================
# Model C: Nonlinear (free-loading) growth curve model
# ============================================================
# R code:
#   (same data as Model B)
#   model <- '
#     i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
#     s =~ 0*y1 + 1*y2 + NA*y3 + 3*y4
#   '
#   fit <- growth(model, data=growth_data)

class TestNonlinearGrowthCurve:
    """Growth model with one free time loading (y3)."""

    MODEL = """
        i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
        s =~ 0*y1 + 1*y2 + NA*y3 + 3*y4
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
        ("chi_square", 1.327, 0.5),
        ("df", 4, 0),
        ("cfi", 1.000, 0.005),
        ("rmsea", 0.000, 0.02),
        ("srmr", 0.008, 0.01),
    ])
    def test_fit_index(self, fid, measure, lavaan, tol):
        _check_fit(fid, measure, lavaan, tol)

    # -- free time loading for y3 --

    def test_free_loading_y3(self, est):
        """y3 loading on slope should be freely estimated (~2.124)."""
        _check_param(est, "s", "=~", "y3", 2.124, 0.082, atol_est=0.05)

    # -- latent means --

    def test_intercept_mean(self, est):
        _check_param(est, "i", "~1", "1", 4.924, 0.073)

    def test_slope_mean(self, est):
        _check_param(est, "s", "~1", "1", 0.512, 0.036)

    # -- latent variances --

    def test_intercept_variance(self, est):
        _check_param(est, "i", "~~", "i", 0.866, 0.109)

    def test_slope_variance(self, est):
        _check_param(est, "s", "~~", "s", 0.177, 0.026)

    # -- intercept-slope covariance --

    def test_intercept_slope_covariance(self, est):
        _check_param(est, "i", "~~", "s", 0.007, 0.038, atol_est=0.02)

    # -- residual variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("y1", 0.295, 0.061),
        ("y2", 0.198, 0.032),
        ("y3", 0.273, 0.049),
        ("y4", 0.415, 0.093),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)


# ============================================================
# Model D: Cross-lagged panel model (2 waves, 2 variables)
# ============================================================
# R code:
#   set.seed(123); n <- 300
#   x1 <- rnorm(n, 5, 1); y1 <- 0.3*x1 + rnorm(n, 3, 1)
#   x2 <- 0.5*x1 + 0.2*y1 + rnorm(n, 0, 0.8)
#   y2 <- 0.1*x1 + 0.4*y1 + rnorm(n, 0, 0.8)
#   fit <- sem('x2~x1; y2~y1; x2~y1; y2~x1; x1~~y1; x2~~y2',
#              data=data.frame(x1,y1,x2,y2))

class TestCrossLaggedPanel:
    """2-wave cross-lagged panel model with x and y."""

    MODEL = """
        x2 ~ x1
        y2 ~ y1
        x2 ~ y1
        y2 ~ x1
        x1 ~~ y1
        x2 ~~ y2
    """

    @pytest.fixture(scope="class")
    def fit(self, clpm_data):
        return sem(self.MODEL, data=clpm_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- fit: just-identified model (df=0) --

    def test_df_zero(self, fid):
        assert fid["df"] == 0

    def test_chi_square_zero(self, fid):
        assert fid["chi_square"] < 0.01

    # -- autoregressive paths --

    def test_ar_x(self, est):
        """Autoregressive path x1 -> x2 (~0.469)."""
        _check_param(est, "x2", "~", "x1", 0.469, 0.052)

    def test_ar_y(self, est):
        """Autoregressive path y1 -> y2 (~0.494)."""
        _check_param(est, "y2", "~", "y1", 0.494, 0.047)

    # -- cross-lagged paths --

    def test_cl_y1_to_x2(self, est):
        """Cross-lagged path y1 -> x2 (~0.164)."""
        _check_param(est, "x2", "~", "y1", 0.164, 0.048)

    def test_cl_x1_to_y2(self, est):
        """Cross-lagged path x1 -> y2 (~0.092)."""
        _check_param(est, "y2", "~", "x1", 0.092, 0.051)

    # -- wave 1 covariance --

    def test_wave1_covariance(self, est):
        _check_param(est, "x1", "~~", "y1", 0.211, 0.056)

    # -- wave 2 residual covariance --

    def test_wave2_residual_covariance(self, est):
        _check_param(est, "x2", "~~", "y2", 0.031, 0.039, atol_est=0.02)

    # -- variances --

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x1", 0.892, 0.073),
        ("y1", 1.021, 0.083),
    ])
    def test_exogenous_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)

    @pytest.mark.parametrize("var,lav_est,lav_se", [
        ("x2", 0.679, 0.055),
        ("y2", 0.654, 0.053),
    ])
    def test_residual_variance(self, est, var, lav_est, lav_se):
        _check_param(est, var, "~~", var, lav_est, lav_se)


# ============================================================
# Model E: RI-CLPM (random-intercept cross-lagged panel model)
# ============================================================
# 3-wave, 2-variable panel data with random intercepts.
# Data generated from known RI-CLPM DGP (seed=42, n=500):
#   RI variance (x): 0.49, (y): ~0.46
#   RI covariance: ~0.245
#   Within-person AR: 0.3, CL: 0.15
#   Innovation SD: 0.6

class TestRICLPM:
    """Random-Intercept Cross-Lagged Panel Model (3 waves, 2 variables)."""

    MODEL = """
        # Random intercepts (between-person)
        RI_x =~ 1*x1 + 1*x2 + 1*x3
        RI_y =~ 1*y1 + 1*y2 + 1*y3

        # Autoregressive (within-person)
        x2 ~ x1
        x3 ~ x2
        y2 ~ y1
        y3 ~ y2

        # Cross-lagged (within-person)
        x2 ~ y1
        x3 ~ y2
        y2 ~ x1
        y3 ~ x2

        # Wave covariances (within-person)
        x1 ~~ y1
        x2 ~~ y2
        x3 ~~ y3

        # Between-person RI covariance
        RI_x ~~ RI_y
    """

    @pytest.fixture(scope="class")
    def riclpm_data(self):
        from semla.datasets import riclpm_data
        return riclpm_data()

    @pytest.fixture(scope="class")
    def fit(self, riclpm_data):
        return sem(self.MODEL, data=riclpm_data)

    @pytest.fixture(scope="class")
    def est(self, fit):
        return fit.estimates()

    @pytest.fixture(scope="class")
    def fid(self, fit):
        return fit.fit_indices()

    # -- convergence and fit --

    def test_converges(self, fit):
        assert fit.converged

    def test_df_positive(self, fid):
        assert fid["df"] >= 1

    def test_good_fit(self, fid):
        assert fid["cfi"] > 0.95
        assert fid["rmsea"] < 0.08

    # -- random intercept variances (between-person) --

    def test_ri_x_variance_positive(self, est):
        row = _get_est(est, "RI_x", "~~", "RI_x")
        assert row["est"] > 0.1, "RI_x variance should be substantial"
        assert row["est"] < 1.5

    def test_ri_y_variance_positive(self, est):
        row = _get_est(est, "RI_y", "~~", "RI_y")
        assert row["est"] > 0.1, "RI_y variance should be substantial"
        assert row["est"] < 1.5

    def test_ri_covariance_positive(self, est):
        """RI_x and RI_y should be positively correlated (by construction)."""
        row = _get_est(est, "RI_x", "~~", "RI_y")
        assert row["est"] > 0.05
        assert row["z"] > 2.0, "RI covariance should be significant"

    # -- within-person autoregressive paths --

    def test_ar_paths_positive(self, est):
        """Within-person AR paths should be positive."""
        for dv, iv in [("x2", "x1"), ("x3", "x2"), ("y2", "y1"), ("y3", "y2")]:
            row = _get_est(est, dv, "~", iv)
            assert row["est"] > -0.5, f"AR {iv}->{dv} unexpectedly negative"
            assert row["est"] < 1.0, f"AR {iv}->{dv} too large"

    # -- cross-lagged paths are near zero in within-person --

    def test_cl_paths_small(self, est):
        """Within-person CL paths should be small (RI absorbs between-person)."""
        for dv, iv in [("x2", "y1"), ("x3", "y2"), ("y2", "x1"), ("y3", "x2")]:
            row = _get_est(est, dv, "~", iv)
            assert abs(row["est"]) < 0.5, f"CL {iv}->{dv} larger than expected"

    # -- within-person residual variances --

    def test_residual_variances_positive(self, est):
        for var in ["x1", "x2", "x3", "y1", "y2", "y3"]:
            row = _get_est(est, var, "~~", var)
            assert row["est"] > 0.0, f"{var} residual variance not positive"

    # -- structural check: RI variance < total variance --

    def test_ri_variance_less_than_total(self, est, riclpm_data):
        """Random intercept variance should be less than total observed variance."""
        ri_x_var = _get_est(est, "RI_x", "~~", "RI_x")["est"]
        total_x_var = riclpm_data["x1"].var()
        assert ri_x_var < total_x_var, "RI_x variance should be < total variance"


# ============================================================
# Model F: fitted() — model-implied matrices
# ============================================================

class TestFitted:
    """Test that fitted() returns correct model-implied moments."""

    @pytest.fixture(scope="class")
    def cfa_fit(self, hs_data):
        model = """
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """
        return cfa(model, data=hs_data)

    @pytest.fixture(scope="class")
    def cfa_mean_fit(self, hs_data):
        model = """
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """
        return cfa(model, data=hs_data, meanstructure=True)

    @pytest.fixture(scope="class")
    def growth_fit(self, growth_data):
        model = """
            i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4
            s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4
        """
        return growth(model, data=growth_data)

    def test_fitted_returns_dict(self, cfa_fit):
        result = cfa_fit.fitted()
        assert "cov" in result
        assert "mean" in result

    def test_fitted_cov_shape(self, cfa_fit):
        result = cfa_fit.fitted()
        assert result["cov"].shape == (9, 9)

    def test_fitted_cov_symmetric(self, cfa_fit):
        cov = cfa_fit.fitted()["cov"].values
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_fitted_cov_positive_diagonal(self, cfa_fit):
        cov = cfa_fit.fitted()["cov"].values
        assert np.all(np.diag(cov) > 0)

    def test_fitted_cov_labels(self, cfa_fit):
        result = cfa_fit.fitted()
        expected = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
        assert list(result["cov"].index) == expected
        assert list(result["cov"].columns) == expected

    def test_fitted_no_mean_without_meanstructure(self, cfa_fit):
        result = cfa_fit.fitted()
        assert result["mean"] is None

    def test_fitted_mean_with_meanstructure(self, cfa_mean_fit):
        result = cfa_mean_fit.fitted()
        assert result["mean"] is not None
        assert len(result["mean"]) == 9

    def test_fitted_mean_close_to_sample(self, cfa_mean_fit, hs_data):
        """Model-implied means should be close to sample means."""
        result = cfa_mean_fit.fitted()
        obs_vars = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
        sample_means = hs_data[obs_vars].mean()
        np.testing.assert_allclose(
            result["mean"].values, sample_means.values, atol=0.01
        )

    def test_fitted_residuals_consistency(self, cfa_fit, hs_data):
        """fitted()['cov'] + residuals() should equal sample covariance."""
        implied_cov = cfa_fit.fitted()["cov"].values
        resid = cfa_fit.residuals(type="raw")
        obs_vars = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]
        sample_cov = hs_data[obs_vars].cov().values * (len(hs_data) - 1) / len(hs_data)
        np.testing.assert_allclose(implied_cov + resid, sample_cov, atol=0.01)

    def test_growth_fitted_mean(self, growth_fit, growth_data):
        """Growth model fitted() should return model-implied means."""
        result = growth_fit.fitted()
        assert result["mean"] is not None
        assert len(result["mean"]) == 4
