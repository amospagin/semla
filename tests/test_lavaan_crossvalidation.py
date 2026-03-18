"""Cross-validation of semla against lavaan reference values.

Compares exact numerical results across multiple models and analysis types.
Reference values from lavaan 0.6-19 documentation and tutorials.
"""

import pytest
from semla import cfa, chi_square_diff_test
from semla.datasets import HolzingerSwineford1939

HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


# ============================================================
# 3-Factor CFA (lavaan tutorial reference)
# ============================================================

class TestThreeFactorCFA:
    """Reference: https://lavaan.ugent.be/tutorial/cfa.html"""

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(HS_MODEL, data=hs_data)

    @pytest.mark.parametrize("measure,lavaan,tol", [
        ("chi_square", 85.306, 0.5),
        ("df", 24, 0),
        ("cfi", 0.931, 0.005),
        ("tli", 0.896, 0.005),
        ("rmsea", 0.092, 0.005),
        ("srmr", 0.065, 0.005),
    ])
    def test_fit_index(self, fit, measure, lavaan, tol):
        semla_val = fit.fit_indices()[measure]
        assert abs(semla_val - lavaan) <= tol, (
            f"{measure}: semla={semla_val:.4f}, lavaan={lavaan}, diff={abs(semla_val-lavaan):.4f}"
        )

    # Loading estimates (lavaan reference values)
    @pytest.mark.parametrize("lv,ind,lav_est,lav_se", [
        ("visual", "x2", 0.554, 0.100),
        ("visual", "x3", 0.729, 0.109),
        ("textual", "x5", 1.113, 0.065),
        ("textual", "x6", 0.926, 0.055),
        ("speed", "x8", 1.180, 0.165),
        ("speed", "x9", 1.082, 0.151),
    ])
    def test_loading(self, fit, lv, ind, lav_est, lav_se):
        est = fit.estimates()
        row = est[(est["lhs"] == lv) & (est["op"] == "=~") & (est["rhs"] == ind)]
        assert abs(row["est"].values[0] - lav_est) < 0.01
        assert abs(row["se"].values[0] - lav_se) < 0.01


# ============================================================
# Multi-Group CFA (lavaan groups tutorial reference)
# ============================================================

class TestMultiGroupCFA:
    """Reference: https://lavaan.ugent.be/tutorial/groups.html"""

    @pytest.fixture(scope="class")
    def fit_config(self, hs_data):
        return cfa(HS_MODEL, data=hs_data, group="school")

    @pytest.fixture(scope="class")
    def fit_metric(self, hs_data):
        return cfa(HS_MODEL, data=hs_data, group="school", invariance="metric")

    def test_configural_chi_square(self, fit_config):
        # lavaan: 115.851
        assert abs(fit_config.fit_indices()["chi_square"] - 115.851) < 1.5

    def test_configural_df(self, fit_config):
        assert fit_config.fit_indices()["df"] == 48

    def test_metric_chi_square(self, fit_metric):
        # lavaan: 124.044
        assert abs(fit_metric.fit_indices()["chi_square"] - 124.044) < 1.5

    def test_metric_df(self, fit_metric):
        assert fit_metric.fit_indices()["df"] == 54

    def test_chi_square_diff(self, fit_config, fit_metric):
        diff = chi_square_diff_test(fit_metric, fit_config)
        # lavaan: chi_sq_diff=8.192, df_diff=6, p=0.224
        assert abs(diff["chi_sq_diff"] - 8.192) < 0.5
        assert diff["df_diff"] == 6
        assert abs(diff["p_value"] - 0.224) < 0.02


# ============================================================
# Standardized solution (lavaan reference for std.all loadings)
# ============================================================

class TestStandardizedSolution:
    """Reference: lavaan standardizedSolution() for HS 3-factor CFA."""

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa(HS_MODEL, data=hs_data)

    @pytest.mark.parametrize("lv,ind,lav_std", [
        ("visual", "x1", 0.772),
        ("visual", "x2", 0.424),
        ("visual", "x3", 0.581),
        ("textual", "x4", 0.852),
        ("textual", "x5", 0.855),
        ("textual", "x6", 0.838),
        ("speed", "x7", 0.570),
        ("speed", "x8", 0.723),
        ("speed", "x9", 0.665),
    ])
    def test_std_all_loading(self, fit, lv, ind, lav_std):
        std = fit.standardized_estimates("std.all")
        row = std[(std["lhs"] == lv) & (std["rhs"] == ind)]
        assert abs(row["est.std"].values[0] - lav_std) < 0.01, (
            f"{lv}=~{ind}: semla={row['est.std'].values[0]:.3f}, lavaan={lav_std}"
        )
