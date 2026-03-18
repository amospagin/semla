"""Tests for standardized parameter estimates."""

import numpy as np
import pandas as pd
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


# lavaan standardized loadings (std.all)
LAVAAN_STD_ALL = [
    ("visual", "x1", 0.772),
    ("visual", "x2", 0.424),
    ("visual", "x3", 0.581),
    ("textual", "x4", 0.852),
    ("textual", "x5", 0.855),
    ("textual", "x6", 0.838),
    ("speed", "x7", 0.570),
    ("speed", "x8", 0.723),
    ("speed", "x9", 0.665),
]


class TestStdAll:
    """Test fully standardized solution (std.all)."""

    @pytest.mark.parametrize(
        "lv,ind,expected",
        LAVAAN_STD_ALL,
        ids=[f"{lv}->{ind}" for lv, ind, _ in LAVAAN_STD_ALL],
    )
    def test_loading(self, hs_fit, lv, ind, expected):
        std = hs_fit.standardized_estimates("std.all")
        row = std[(std["lhs"] == lv) & (std["rhs"] == ind)]
        assert abs(row["est.std"].values[0] - expected) < 0.01

    def test_latent_variances_are_one(self, hs_fit):
        """Latent variances should be 1.0 in std.all."""
        std = hs_fit.standardized_estimates("std.all")
        for lv in ["visual", "textual", "speed"]:
            row = std[(std["lhs"] == lv) & (std["op"] == "~~") & (std["rhs"] == lv)]
            assert abs(row["est.std"].values[0] - 1.0) < 0.001

    def test_latent_covariances_are_correlations(self, hs_fit):
        """Standardized latent covariances should be between -1 and 1."""
        std = hs_fit.standardized_estimates("std.all")
        covs = std[(std["op"] == "~~") & (std["lhs"] != std["rhs"])]
        assert (covs["est.std"].abs() <= 1.0 + 1e-6).all()

    def test_residual_variance_plus_r2(self, hs_fit):
        """Standardized residual variance + loading^2 should approximate 1.0."""
        std = hs_fit.standardized_estimates("std.all")
        for lv, ind, _ in LAVAAN_STD_ALL:
            loading = std[(std["lhs"] == lv) & (std["rhs"] == ind)]["est.std"].values[0]
            # This only holds exactly for single-factor indicators
            # For the HS model with no cross-loadings it should be close
            resid = std[(std["lhs"] == ind) & (std["op"] == "~~") & (std["rhs"] == ind)]["est.std"].values[0]
            # loading^2 + residual should be ~1.0
            total = loading ** 2 + resid
            assert abs(total - 1.0) < 0.01, f"{ind}: loading^2 + resid = {total:.3f}"


class TestStdLv:
    """Test standardization by LV SD only (std.lv)."""

    def test_returns_dataframe(self, hs_fit):
        std = hs_fit.standardized_estimates("std.lv")
        assert isinstance(std, pd.DataFrame)
        assert "est.std" in std.columns

    def test_loadings_scaled_by_lv_sd(self, hs_fit):
        """std.lv loadings should equal est * sqrt(latent_variance)."""
        std = hs_fit.standardized_estimates("std.lv")
        est = hs_fit.estimates()
        for lv in ["visual", "textual", "speed"]:
            lv_var = est[(est["lhs"] == lv) & (est["op"] == "~~") & (est["rhs"] == lv)]["est"].values[0]
            lv_sd = np.sqrt(lv_var)
            loadings = std[(std["lhs"] == lv) & (std["op"] == "=~")]
            for _, row in loadings.iterrows():
                expected = row["est"] * lv_sd
                assert abs(row["est.std"] - expected) < 0.001


class TestInvalidType:
    def test_invalid_type_raises(self, hs_fit):
        with pytest.raises(ValueError, match="type must be"):
            hs_fit.standardized_estimates("invalid")
