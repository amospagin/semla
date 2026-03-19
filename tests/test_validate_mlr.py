"""Validate MLR standard errors and fit indices against lavaan reference values.

Reference: lavaan 0.6-21, HolzingerSwineford1939 3-factor CFA model,
           estimator = "MLR".
Source: parameterEstimates(fit) and fitMeasures(fit) from lavaan.

GitHub issue: #21
"""

import numpy as np
import pandas as pd
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_mlr_fit():
    """Fit the classic 3-factor CFA with MLR estimator."""
    df = HolzingerSwineford1939()
    model = """
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """
    return cfa(model, data=df, estimator="MLR")


# lavaan 0.6-21 MLR reference values: (lhs, op, rhs, estimate, robust_se)
LAVAAN_MLR_PARAMS = [
    # Factor loadings (free only; marker indicators fixed to 1.0)
    ("visual", "=~", "x2", 0.553500, 0.132078),
    ("visual", "=~", "x3", 0.729370, 0.141087),
    ("textual", "=~", "x5", 1.113077, 0.065686),
    ("textual", "=~", "x6", 0.926146, 0.061378),
    ("speed", "=~", "x8", 1.179951, 0.130445),
    ("speed", "=~", "x9", 1.081530, 0.266376),
    # Residual variances
    ("x1", "~~", "x1", 0.549054, 0.156468),
    ("x2", "~~", "x2", 1.133839, 0.111879),
    ("x3", "~~", "x3", 0.844324, 0.100286),
    ("x4", "~~", "x4", 0.371173, 0.050282),
    ("x5", "~~", "x5", 0.446255, 0.056704),
    ("x6", "~~", "x6", 0.356203, 0.046517),
    ("x7", "~~", "x7", 0.799392, 0.097222),
    ("x8", "~~", "x8", 0.487697, 0.119533),
    ("x9", "~~", "x9", 0.566131, 0.118739),
    # Latent variances
    ("visual", "~~", "visual", 0.809316, 0.180396),
    ("textual", "~~", "textual", 0.979491, 0.121298),
    ("speed", "~~", "speed", 0.383748, 0.106705),
    # Latent covariances
    ("visual", "~~", "textual", 0.408232, 0.099317),
    ("visual", "~~", "speed", 0.262225, 0.060060),
    ("textual", "~~", "speed", 0.173495, 0.056307),
]

_SE_XFAIL = set()  # All SEs now match lavaan (fixed in sandwich estimator rewrite)


class TestMLRParameterEstimates:
    """Compare MLR parameter estimates against lavaan (tolerance: 0.01).

    MLR uses the same point estimates as ML; only SEs and chi-square differ.
    """

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_est,lav_se",
        LAVAAN_MLR_PARAMS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_MLR_PARAMS],
    )
    def test_estimate(self, hs_mlr_fit, lhs, op, rhs, lav_est, lav_se):
        est = hs_mlr_fit.estimates()
        row = est[(est["lhs"] == lhs) & (est["op"] == op) & (est["rhs"] == rhs)]
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found"
        np.testing.assert_allclose(
            row["est"].values[0], lav_est, atol=0.01,
            err_msg=f"{lhs} {op} {rhs}: est={row['est'].values[0]:.6f}, lavaan={lav_est:.6f}",
        )


class TestMLRRobustStandardErrors:
    """Compare MLR robust SEs against lavaan (tolerance: 0.01).

    Several parameters are marked xfail due to known discrepancies in the
    sandwich estimator implementation (see GitHub issue #21).
    """

    @pytest.mark.parametrize(
        "lhs,op,rhs,lav_est,lav_se",
        LAVAAN_MLR_PARAMS,
        ids=[f"{l} {o} {r}" for l, o, r, _, _ in LAVAAN_MLR_PARAMS],
    )
    def test_robust_se(self, hs_mlr_fit, lhs, op, rhs, lav_est, lav_se):
        param_id = f"{lhs} {op} {rhs}"
        if param_id in _SE_XFAIL:
            pytest.xfail(f"Known MLR SE discrepancy for {param_id} (issue #21)")

        est = hs_mlr_fit.estimates()
        row = est[(est["lhs"] == lhs) & (est["op"] == op) & (est["rhs"] == rhs)]
        assert len(row) == 1, f"Parameter {lhs} {op} {rhs} not found"
        np.testing.assert_allclose(
            row["se"].values[0], lav_se, atol=0.01,
            err_msg=f"{lhs} {op} {rhs}: se={row['se'].values[0]:.6f}, lavaan={lav_se:.6f}",
        )


class TestMLRFitIndices:
    """Compare MLR Satorra-Bentler scaled fit indices against lavaan."""

    def test_satorra_bentler_chi_square(self, hs_mlr_fit):
        # lavaan chisq.scaled = 87.131603 (yuan.bentler.mplus)
        idx = hs_mlr_fit.fit_indices()
        np.testing.assert_allclose(idx["chi_square"], 87.131603, atol=1.0)

    def test_df(self, hs_mlr_fit):
        assert hs_mlr_fit.fit_indices()["df"] == 24

    def test_cfi(self, hs_mlr_fit):
        # lavaan cfi.scaled = 0.925207
        idx = hs_mlr_fit.fit_indices()
        np.testing.assert_allclose(idx["cfi"], 0.925207, atol=0.005)

    def test_tli(self, hs_mlr_fit):
        # lavaan tli.scaled = 0.887810
        idx = hs_mlr_fit.fit_indices()
        np.testing.assert_allclose(idx["tli"], 0.887810, atol=0.01)

    def test_rmsea(self, hs_mlr_fit):
        # lavaan rmsea.scaled = 0.093483
        idx = hs_mlr_fit.fit_indices()
        np.testing.assert_allclose(idx["rmsea"], 0.093483, atol=0.005)

    def test_srmr(self, hs_mlr_fit):
        # lavaan srmr = 0.065205 (SRMR is not affected by MLR scaling)
        idx = hs_mlr_fit.fit_indices()
        np.testing.assert_allclose(idx["srmr"], 0.065205, atol=0.005)


class TestMLRFixedLoadings:
    """Check that marker indicator loadings are fixed to 1.0 under MLR."""

    def test_first_loadings_fixed(self, hs_mlr_fit):
        est = hs_mlr_fit.estimates()
        for lv, ind in [("visual", "x1"), ("textual", "x4"), ("speed", "x7")]:
            row = est[(est["lhs"] == lv) & (est["op"] == "=~") & (est["rhs"] == ind)]
            assert row["est"].values[0] == 1.0
            # Fixed params may have NaN or 0 for SE
            se_val = row["se"].values[0]
            assert np.isnan(se_val) or se_val == 0.0
