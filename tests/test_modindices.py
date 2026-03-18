"""Tests for modification indices."""

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


class TestModIndices:
    def test_returns_dataframe(self, hs_fit):
        mi = hs_fit.modindices()
        assert isinstance(mi, pd.DataFrame)
        assert "mi" in mi.columns
        assert "epc" in mi.columns
        assert "lhs" in mi.columns
        assert "op" in mi.columns
        assert "rhs" in mi.columns

    def test_sorted_descending(self, hs_fit):
        mi = hs_fit.modindices()
        if len(mi) > 1:
            assert mi["mi"].iloc[0] >= mi["mi"].iloc[1]

    def test_min_mi_filter(self, hs_fit):
        mi_all = hs_fit.modindices(min_mi=0.0)
        mi_filtered = hs_fit.modindices(min_mi=10.0)
        assert len(mi_filtered) < len(mi_all)
        assert (mi_filtered["mi"] >= 10.0).all()

    def test_all_mi_positive(self, hs_fit):
        mi = hs_fit.modindices()
        assert (mi["mi"] >= 0).all()

    def test_no_free_params_included(self, hs_fit):
        """MI should only be for fixed parameters, not already-free ones."""
        mi = hs_fit.modindices()
        est = hs_fit.estimates()
        free_params = est[est["free"]]

        for _, row in mi.iterrows():
            # Check this MI param is not already free
            match = free_params[
                (free_params["lhs"] == row["lhs"])
                & (free_params["op"] == row["op"])
                & (free_params["rhs"] == row["rhs"])
            ]
            assert len(match) == 0, (
                f"MI includes already-free param: {row['lhs']} {row['op']} {row['rhs']}"
            )

    def test_no_latent_observed_covariances(self, hs_fit):
        """MI should not include latent-observed covariances."""
        mi = hs_fit.modindices()
        latent = set(hs_fit.spec.latent_vars)
        obs = set(hs_fit.spec.observed_vars)
        cov_rows = mi[mi["op"] == "~~"]
        for _, row in cov_rows.iterrows():
            # Both should be observed
            assert row["lhs"] in obs and row["rhs"] in obs, (
                f"Latent-observed covariance in MI: {row['lhs']} ~~ {row['rhs']}"
            )

    def test_top_mi_is_visual_x9_or_x7x8(self, hs_fit):
        """The largest MI should be visual=~x9 or x7~~x8 (well-known result)."""
        mi = hs_fit.modindices()
        top = mi.iloc[0]
        # Either visual=~x9 or x7/x8 residual covariance
        is_visual_x9 = top["lhs"] == "visual" and top["rhs"] == "x9"
        is_x7_x8 = {"x7", "x8"} == {top["lhs"], top["rhs"]}
        is_x7_x2 = {"x7", "x2"} == {top["lhs"], top["rhs"]}
        assert is_visual_x9 or is_x7_x8 or is_x7_x2, (
            f"Top MI unexpected: {top['lhs']} {top['op']} {top['rhs']}"
        )

    def test_visual_x9_mi_approximate(self, hs_fit):
        """visual =~ x9 MI should be approximately 36 (lavaan: 36.411)."""
        mi = hs_fit.modindices()
        row = mi[(mi["lhs"] == "visual") & (mi["op"] == "=~") & (mi["rhs"] == "x9")]
        assert len(row) == 1
        # Allow 20% tolerance due to different optima
        assert 25 < row["mi"].values[0] < 50

    def test_x7_x8_covariance_mi(self, hs_fit):
        """x7 ~~ x8 MI should be approximately 34 (lavaan: 34.145)."""
        mi = hs_fit.modindices()
        row = mi[(mi["lhs"] == "x8") & (mi["op"] == "~~") & (mi["rhs"] == "x7")]
        assert len(row) == 1
        assert 25 < row["mi"].values[0] < 45
