"""Tests for equality constraints via parameter labels."""

import numpy as np
import pytest
from semla import cfa
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


class TestEqualityConstraints:
    def test_constrained_loadings_equal(self, hs_data):
        """Same label should produce identical estimates."""
        fit = cfa("visual =~ x1 + a*x2 + a*x3", data=hs_data)
        est = fit.estimates()
        x2 = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        x3 = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        assert abs(x2 - x3) < 1e-8

    def test_constraint_reduces_n_free(self, hs_data):
        """One equality constraint should reduce n_free by 1."""
        fit_free = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_constrained = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        assert fit_constrained.spec.n_free == fit_free.spec.n_free - 1

    def test_constraint_increases_df(self, hs_data):
        """One equality constraint should increase df by 1."""
        fit_free = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_constrained = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        assert fit_constrained.fit_indices()["df"] == fit_free.fit_indices()["df"] + 1

    def test_multiple_constraint_groups(self, hs_data):
        """Multiple independent constraint groups should all be enforced."""
        fit = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + b*x5 + b*x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        est = fit.estimates()
        x2 = est[(est["rhs"] == "x2") & (est["op"] == "=~")]["est"].values[0]
        x3 = est[(est["rhs"] == "x3") & (est["op"] == "=~")]["est"].values[0]
        x5 = est[(est["rhs"] == "x5") & (est["op"] == "=~")]["est"].values[0]
        x6 = est[(est["rhs"] == "x6") & (est["op"] == "=~")]["est"].values[0]

        assert abs(x2 - x3) < 1e-8
        assert abs(x5 - x6) < 1e-8
        # But the two groups should differ
        assert abs(x2 - x5) > 0.01

    def test_constrained_chi_square_higher(self, hs_data):
        """Constrained model should have equal or higher chi-square."""
        fit_free = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_constrained = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        assert fit_constrained.fit_indices()["chi_square"] >= fit_free.fit_indices()["chi_square"] - 0.1

    def test_no_label_no_constraint(self, hs_data):
        """Without labels, no constraints should be applied."""
        fit = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
        """, data=hs_data)
        assert fit.spec._constraint_map is None

    def test_converges(self, hs_data):
        fit = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + b*x5 + b*x6
            speed   =~ x7 + c*x8 + c*x9
        """, data=hs_data)
        assert fit.converged

    def test_chi_square_diff_test_with_constraints(self, hs_data):
        """Can compare constrained vs unconstrained with chi-sq diff test."""
        from semla import chi_square_diff_test

        fit_free = cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        fit_constrained = cfa("""
            visual  =~ x1 + a*x2 + a*x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

        diff = chi_square_diff_test(fit_constrained, fit_free)
        assert diff["df_diff"] == 1
        assert diff["chi_sq_diff"] > 0
        assert 0 <= diff["p_value"] <= 1
