"""Validate semla output against lavaan reference values.

Reference: lavaan 0.6-19, HolzingerSwineford1939 3-factor CFA model.
Source: https://lavaan.ugent.be/tutorial/cfa.html
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_fit():
    """Fit the classic 3-factor CFA on Holzinger-Swineford data."""
    df = HolzingerSwineford1939()
    model = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    return cfa(model, data=df)


class TestFitIndices:
    """Compare fit indices against lavaan."""

    def test_chi_square(self, hs_fit):
        # lavaan: 85.306
        assert abs(hs_fit.fit_indices()["chi_square"] - 85.306) < 0.5

    def test_df(self, hs_fit):
        assert hs_fit.fit_indices()["df"] == 24

    def test_cfi(self, hs_fit):
        # lavaan: 0.931
        assert abs(hs_fit.fit_indices()["cfi"] - 0.931) < 0.005

    def test_tli(self, hs_fit):
        # lavaan: 0.896
        assert abs(hs_fit.fit_indices()["tli"] - 0.896) < 0.005

    def test_rmsea(self, hs_fit):
        # lavaan: 0.092
        assert abs(hs_fit.fit_indices()["rmsea"] - 0.092) < 0.005

    def test_rmsea_ci_lower(self, hs_fit):
        # lavaan: 0.071
        assert abs(hs_fit.fit_indices()["rmsea_ci_lower"] - 0.071) < 0.005

    def test_rmsea_ci_upper(self, hs_fit):
        # lavaan: 0.114
        assert abs(hs_fit.fit_indices()["rmsea_ci_upper"] - 0.114) < 0.005

    def test_srmr(self, hs_fit):
        # lavaan: 0.065
        assert abs(hs_fit.fit_indices()["srmr"] - 0.065) < 0.005


# lavaan reference values: (lhs, op, rhs, estimate, std.err)
LAVAAN_PARAMS = [
    # Factor loadings (free only)
    ("visual", "=~", "x2", 0.554, 0.100),
    ("visual", "=~", "x3", 0.729, 0.109),
    ("textual", "=~", "x5", 1.113, 0.065),
    ("textual", "=~", "x6", 0.926, 0.055),
    ("speed", "=~", "x8", 1.180, 0.165),
    ("speed", "=~", "x9", 1.082, 0.151),
    # Residual variances
    ("x1", "~~", "x1", 0.549, 0.114),
    ("x2", "~~", "x2", 1.134, 0.102),
    ("x3", "~~", "x3", 0.844, 0.091),
    ("x4", "~~", "x4", 0.371, 0.048),
    ("x5", "~~", "x5", 0.446, 0.058),
    ("x6", "~~", "x6", 0.356, 0.043),
    ("x7", "~~", "x7", 0.799, 0.081),
    ("x8", "~~", "x8", 0.488, 0.074),
    ("x9", "~~", "x9", 0.566, 0.071),
    # Latent variances
    ("visual", "~~", "visual", 0.809, 0.145),
    ("textual", "~~", "textual", 0.979, 0.112),
    ("speed", "~~", "speed", 0.384, 0.086),
    # Latent covariances
    ("visual", "~~", "textual", 0.408, 0.074),
    ("visual", "~~", "speed", 0.262, 0.056),
    ("textual", "~~", "speed", 0.173, 0.049),
]


class TestParameterEstimates:
    """Compare parameter estimates against lavaan (tolerance: 0.01)."""

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_est,lav_se",
        LAVAAN_PARAMS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_PARAMS],
    )
    def test_estimate(self, hs_fit, lhs, op, rhs, lav_est, lav_se):
        est = hs_fit.estimates()
        row = est[(est["lhs"] == lhs) & (est["op"] == op) & (est["rhs"] == rhs)]
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found"
        assert abs(row["est"].values[0] - lav_est) < 0.01, (
            f"{lhs} {op} {rhs}: est={row['est'].values[0]:.4f}, lavaan={lav_est}"
        )

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_est,lav_se",
        LAVAAN_PARAMS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_PARAMS],
    )
    def test_standard_error(self, hs_fit, lhs, op, rhs, lav_est, lav_se):
        est = hs_fit.estimates()
        row = est[(est["lhs"] == lhs) & (est["op"] == op) & (est["rhs"] == rhs)]
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found"
        assert abs(row["se"].values[0] - lav_se) < 0.01, (
            f"{lhs} {op} {rhs}: se={row['se'].values[0]:.4f}, lavaan={lav_se}"
        )


class TestFixedLoadings:
    """Check that fixed loadings are correctly set to 1.0."""

    def test_first_loadings_fixed(self, hs_fit):
        est = hs_fit.estimates()
        for lv, ind in [("visual", "x1"), ("textual", "x4"), ("speed", "x7")]:
            row = est[(est["lhs"] == lv) & (est["op"] == "=~") & (est["rhs"] == ind)]
            assert row["est"].values[0] == 1.0
            assert row["free"].values[0] is False or row["free"].values[0] == False
