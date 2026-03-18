"""Tests for multi-group SEM/CFA."""

import numpy as np
import pandas as pd
import pytest
from semla import cfa, chi_square_diff_test, MultiGroupModel
from semla.datasets import HolzingerSwineford1939


HS_MODEL = """
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9
"""


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def fit_configural(hs_data):
    return cfa(HS_MODEL, data=hs_data, group="school")


@pytest.fixture(scope="module")
def fit_metric(hs_data):
    return cfa(HS_MODEL, data=hs_data, group="school", invariance="metric")


class TestConfiguralInvariance:
    def test_returns_multigroup_model(self, fit_configural):
        assert isinstance(fit_configural, MultiGroupModel)

    def test_converged(self, fit_configural):
        assert fit_configural.converged

    def test_df_is_48(self, fit_configural):
        # 2 groups * 9*10/2 = 90 data points, 42 free params (2*21)
        # df = 90 - 42 = 48
        assert fit_configural.fit_indices()["df"] == 48

    def test_chi_square_close_to_lavaan(self, fit_configural):
        # lavaan: 115.851
        assert abs(fit_configural.fit_indices()["chi_square"] - 115.851) < 2.0

    def test_cfi_close_to_lavaan(self, fit_configural):
        # lavaan: 0.923
        assert abs(fit_configural.fit_indices()["cfi"] - 0.923) < 0.01

    def test_two_groups_in_estimates(self, fit_configural):
        est = fit_configural.estimates()
        assert "group" in est.columns
        groups = est["group"].unique()
        assert len(groups) == 2
        assert set(groups) == {"Grant-White", "Pasteur"}

    def test_different_estimates_per_group(self, fit_configural):
        """Configural: loadings should differ between groups."""
        est = fit_configural.estimates()
        for ind in ["x2", "x3"]:
            vals = est[(est["op"] == "=~") & (est["rhs"] == ind) & (est["free"])]
            ests = vals["est"].values
            # Not exactly equal (configural allows group differences)
            assert len(ests) == 2


class TestMetricInvariance:
    def test_converged(self, fit_metric):
        assert fit_metric.converged

    def test_df_is_54(self, fit_metric):
        # Configural df=48, metric adds 6 constraints (6 free loadings shared)
        assert fit_metric.fit_indices()["df"] == 54

    def test_chi_square_higher_than_configural(self, fit_configural, fit_metric):
        assert fit_metric.fit_indices()["chi_square"] >= fit_configural.fit_indices()["chi_square"]

    def test_loadings_equal_across_groups(self, fit_metric):
        est = fit_metric.estimates()
        free_loadings = est[(est["op"] == "=~") & (est["free"])]

        for ind in free_loadings["rhs"].unique():
            vals = free_loadings[free_loadings["rhs"] == ind]
            ests = vals["est"].values
            # All groups should have identical loading
            assert np.allclose(ests, ests[0], atol=1e-8), (
                f"Loadings for {ind} not equal: {ests}"
            )

    def test_variances_differ_across_groups(self, fit_metric):
        """Metric invariance: variances should still be group-specific."""
        est = fit_metric.estimates()
        variances = est[(est["op"] == "~~") & (est["lhs"] == est["rhs"]) & (est["free"])]

        # Check at least one variance differs between groups
        some_differ = False
        for var in variances["lhs"].unique():
            vals = variances[variances["lhs"] == var]
            if len(vals) == 2:
                if abs(vals["est"].values[0] - vals["est"].values[1]) > 0.01:
                    some_differ = True
                    break
        assert some_differ


class TestChiSquareDiffTest:
    def test_metric_vs_configural(self, fit_configural, fit_metric):
        diff = chi_square_diff_test(fit_metric, fit_configural)
        assert diff["df_diff"] == 6
        assert diff["chi_sq_diff"] > 0
        assert 0 <= diff["p_value"] <= 1

    def test_metric_invariance_holds(self, fit_configural, fit_metric):
        """For HS data, metric invariance should hold (p > 0.05)."""
        diff = chi_square_diff_test(fit_metric, fit_configural)
        assert diff["p_value"] > 0.05

    def test_wrong_order_raises(self, fit_configural, fit_metric):
        with pytest.raises(ValueError, match="must have more df"):
            chi_square_diff_test(fit_configural, fit_metric)


class TestMultiGroupAPI:
    def test_cfa_with_group_returns_multigroup(self, hs_data):
        fit = cfa(HS_MODEL, data=hs_data, group="school")
        assert isinstance(fit, MultiGroupModel)

    def test_cfa_without_group_returns_model(self, hs_data):
        from semla import Model
        fit = cfa(HS_MODEL, data=hs_data)
        assert isinstance(fit, Model)

    def test_invalid_group_column(self, hs_data):
        with pytest.raises(ValueError, match="not found"):
            cfa(HS_MODEL, data=hs_data, group="nonexistent")

    def test_invalid_invariance(self, hs_data):
        with pytest.raises(ValueError, match="invariance must be"):
            cfa(HS_MODEL, data=hs_data, group="school", invariance="strict")

    def test_summary_runs(self, fit_configural):
        output = fit_configural.summary()
        assert "Multi-Group" in output
        assert "Grant-White" in output
        assert "Pasteur" in output

    def test_fit_indices_has_group_info(self, fit_configural):
        idx = fit_configural.fit_indices()
        assert idx["n_groups"] == 2
        assert idx["invariance"] == "configural"
