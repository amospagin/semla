"""Tests for bootstrap confidence intervals."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_fit():
    return cfa("""
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9
    """, data=HolzingerSwineford1939())


@pytest.fixture(scope="module")
def boot_results(hs_fit):
    return hs_fit.bootstrap(nboot=50, seed=42)


class TestBootstrap:
    def test_returns_dataframe(self, boot_results):
        assert "se_boot" in boot_results.columns
        assert "ci.lower" in boot_results.columns
        assert "ci.upper" in boot_results.columns

    def test_correct_number_of_rows(self, hs_fit, boot_results):
        n_free = len(hs_fit.estimates()[hs_fit.estimates()["free"]])
        assert len(boot_results) == n_free

    def test_ci_contains_estimate(self, boot_results):
        """Point estimate should be within bootstrap CI."""
        for _, row in boot_results.iterrows():
            if not np.isnan(row["ci.lower"]):
                assert row["ci.lower"] <= row["est"] <= row["ci.upper"], (
                    f"{row['lhs']} {row['op']} {row['rhs']}: "
                    f"est={row['est']:.3f} not in [{row['ci.lower']:.3f}, {row['ci.upper']:.3f}]"
                )

    def test_se_boot_positive(self, boot_results):
        assert (boot_results["se_boot"] > 0).all()

    def test_most_replications_valid(self, boot_results):
        assert (boot_results["n_valid"] >= 40).all()

    def test_reproducible_with_seed(self, hs_fit):
        b1 = hs_fit.bootstrap(nboot=20, seed=123)
        b2 = hs_fit.bootstrap(nboot=20, seed=123)
        np.testing.assert_allclose(b1["se_boot"].values, b2["se_boot"].values)
