"""Tests for mean structure (intercepts)."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


@pytest.fixture(scope="module")
def fit_no_means(hs_data):
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=hs_data)


@pytest.fixture(scope="module")
def fit_with_means(hs_data):
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=hs_data, meanstructure=True)


class TestMeanStructureBasic:
    def test_converged(self, fit_with_means):
        assert fit_with_means.converged

    def test_same_chi_square_as_no_means(self, fit_no_means, fit_with_means):
        """Adding saturated mean structure shouldn't change chi-square."""
        chi1 = fit_no_means.fit_indices()["chi_square"]
        chi2 = fit_with_means.fit_indices()["chi_square"]
        assert abs(chi1 - chi2) < 0.5

    def test_same_df(self, fit_no_means, fit_with_means):
        """df should be same: extra data points = extra parameters."""
        assert fit_no_means.fit_indices()["df"] == fit_with_means.fit_indices()["df"]

    def test_intercepts_in_estimates(self, fit_with_means):
        est = fit_with_means.estimates()
        intercepts = est[est["op"] == "~1"]
        assert len(intercepts) == 9  # one per observed variable

    def test_intercepts_equal_sample_means(self, fit_with_means, hs_data):
        """With saturated mean structure, intercepts should equal sample means."""
        est = fit_with_means.estimates()
        intercepts = est[est["op"] == "~1"]
        for _, row in intercepts.iterrows():
            sample_mean = hs_data[row["lhs"]].mean()
            assert abs(row["est"] - sample_mean) < 0.01, (
                f"Intercept for {row['lhs']}: {row['est']:.3f} != sample mean {sample_mean:.3f}"
            )

    def test_intercepts_have_se(self, fit_with_means):
        est = fit_with_means.estimates()
        intercepts = est[(est["op"] == "~1") & (est["free"])]
        assert not intercepts["se"].isna().all(), "All intercept SEs are NaN"

    def test_loadings_unchanged(self, fit_no_means, fit_with_means):
        """Factor loadings should be the same with/without mean structure."""
        est1 = fit_no_means.estimates()
        est2 = fit_with_means.estimates()
        load1 = est1[(est1["op"] == "=~") & (est1["free"])]["est"].values
        load2 = est2[(est2["op"] == "=~") & (est2["free"])]["est"].values
        np.testing.assert_allclose(load1, load2, atol=0.01)

    def test_summary_includes_intercepts(self, fit_with_means):
        summary = fit_with_means.summary()
        assert "Intercepts:" in summary

    def test_no_latent_intercepts(self, fit_with_means):
        """Latent variable intercepts should be fixed to 0 (not in estimates)."""
        est = fit_with_means.estimates()
        intercepts = est[est["op"] == "~1"]
        latent_intercepts = intercepts[intercepts["lhs"].isin(["visual", "textual", "speed"])]
        assert len(latent_intercepts) == 0


class TestMeanStructureAutoDetect:
    def test_tilde1_activates_mean_structure(self, hs_data):
        """Using ~1 in syntax should auto-enable mean structure."""
        fit = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
            x1 ~1
        """, data=hs_data)
        est = fit.estimates()
        # Should have intercepts for ALL observed vars (auto-added)
        intercepts = est[est["op"] == "~1"]
        assert len(intercepts) == 9


class TestBackwardCompatibility:
    def test_default_no_mean_structure(self, fit_no_means):
        """By default, no intercepts should appear."""
        est = fit_no_means.estimates()
        intercepts = est[est["op"] == "~1"]
        assert len(intercepts) == 0

    def test_spec_meanstructure_false(self, fit_no_means):
        assert not fit_no_means.spec.meanstructure
