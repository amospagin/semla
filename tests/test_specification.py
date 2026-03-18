"""Tests for the RAM specification builder."""

import numpy as np
import pytest
from semla.specification import build_specification
from semla.syntax import parse_syntax


class TestBuildSpecification:
    def test_basic_cfa_structure(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
        """
        tokens = parse_syntax(model)
        obs = ["x1", "x2", "x3", "x4", "x5", "x6"]
        spec = build_specification(tokens, obs)

        assert spec.observed_vars == obs
        assert spec.latent_vars == ["f1", "f2"]
        assert spec.n_obs == 6
        assert spec.n_latent == 2
        assert spec.n_vars == 8

    def test_filter_matrix(self):
        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        obs = ["x1", "x2", "x3"]
        spec = build_specification(tokens, obs)

        # F should be (3, 4) — 3 observed, 4 total (3 obs + 1 latent)
        assert spec.F.shape == (3, 4)
        # First 3 columns should be identity
        np.testing.assert_array_equal(spec.F[:, :3], np.eye(3))
        # Last column should be zeros
        np.testing.assert_array_equal(spec.F[:, 3], [0, 0, 0])

    def test_first_loading_fixed(self):
        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        obs = ["x1", "x2", "x3"]
        spec = build_specification(tokens, obs)

        # A[x1, f1] should be fixed at 1.0
        i_x1 = spec.all_vars.index("x1")
        i_f1 = spec.all_vars.index("f1")
        assert not spec.A_free[i_x1, i_f1]
        assert spec.A_values[i_x1, i_f1] == 1.0

        # A[x2, f1] should be free
        i_x2 = spec.all_vars.index("x2")
        assert spec.A_free[i_x2, i_f1]

    def test_auto_residual_variances(self):
        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        obs = ["x1", "x2", "x3"]
        spec = build_specification(tokens, obs)

        # All variables should have free variance on the diagonal of S
        for var in spec.all_vars:
            i = spec.all_vars.index(var)
            assert spec.S_free[i, i], f"Variance of {var} should be free"

    def test_auto_latent_covariances(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
        """
        tokens = parse_syntax(model)
        obs = ["x1", "x2", "x3", "x4", "x5", "x6"]
        spec = build_specification(tokens, obs, auto_cov_latent=True)

        i_f1 = spec.all_vars.index("f1")
        i_f2 = spec.all_vars.index("f2")
        assert spec.S_free[i_f1, i_f2]
        assert spec.S_free[i_f2, i_f1]

    def test_no_auto_latent_cov_when_disabled(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
        """
        tokens = parse_syntax(model)
        obs = ["x1", "x2", "x3", "x4", "x5", "x6"]
        spec = build_specification(tokens, obs, auto_cov_latent=False)

        i_f1 = spec.all_vars.index("f1")
        i_f2 = spec.all_vars.index("f2")
        assert not spec.S_free[i_f1, i_f2]

    def test_pack_unpack_roundtrip(self):
        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        obs = ["x1", "x2", "x3"]
        spec = build_specification(tokens, obs)

        theta = spec.pack_start()
        A, S = spec.unpack(theta)

        # Should be valid matrices
        assert A.shape == (4, 4)
        assert S.shape == (4, 4)
        # S should be symmetric
        np.testing.assert_array_almost_equal(S, S.T)

    def test_free_param_count_cfa(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
        """
        tokens = parse_syntax(model)
        obs = ["x1", "x2", "x3", "x4", "x5", "x6"]
        spec = build_specification(tokens, obs, auto_cov_latent=True)

        # Free params in A: 4 loadings (2 fixed per factor)
        # Free params in S: 8 variances + 2 covariance cells (symmetric) = 10
        # Total: 4 + 10 = 14
        assert spec.n_free == 14

    def test_regression_paths(self):
        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
            f2 ~ f1
        """
        tokens = parse_syntax(model)
        obs = ["x1", "x2", "x3", "x4", "x5", "x6"]
        spec = build_specification(tokens, obs, auto_cov_latent=False)

        i_f1 = spec.all_vars.index("f1")
        i_f2 = spec.all_vars.index("f2")
        assert spec.A_free[i_f2, i_f1]
