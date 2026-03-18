"""Tests for ML estimation."""

import numpy as np
import pandas as pd
import pytest
from semla.estimation import estimate, ml_objective, _model_implied_cov
from semla.specification import build_specification
from semla.syntax import parse_syntax


class TestModelImpliedCov:
    def test_simple_case(self):
        """With A=0, model-implied cov should equal S (filtered)."""
        n = 3
        A = np.zeros((n, n))
        S = np.diag([1.0, 2.0, 3.0])
        F = np.eye(n)

        sigma = _model_implied_cov(A, S, F)
        np.testing.assert_array_almost_equal(sigma, S)

    def test_with_latent(self):
        """One factor with two indicators."""
        # all_vars: [x1, x2, f1]
        A = np.zeros((3, 3))
        A[0, 2] = 1.0  # x1 <- f1 (loading=1)
        A[1, 2] = 0.8  # x2 <- f1 (loading=0.8)

        S = np.zeros((3, 3))
        S[0, 0] = 0.5  # residual var x1
        S[1, 1] = 0.5  # residual var x2
        S[2, 2] = 1.0  # latent var f1

        F = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

        sigma = _model_implied_cov(A, S, F)
        assert sigma.shape == (2, 2)

        # Var(x1) = 1^2 * 1.0 + 0.5 = 1.5
        assert abs(sigma[0, 0] - 1.5) < 1e-10
        # Var(x2) = 0.8^2 * 1.0 + 0.5 = 1.14
        assert abs(sigma[1, 1] - 1.14) < 1e-10
        # Cov(x1, x2) = 1 * 0.8 * 1.0 = 0.8
        assert abs(sigma[0, 1] - 0.8) < 1e-10


class TestMLObjective:
    def test_perfect_fit_is_zero(self):
        """When model-implied cov equals sample cov, F_ML should be 0."""
        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        obs = ["x1", "x2", "x3"]
        spec = build_specification(tokens, obs)

        # Construct a covariance matrix consistent with the model
        A = spec.A_values.copy()
        A[1, 3] = 0.7  # x2 loading
        A[2, 3] = 0.9  # x3 loading
        S = np.diag([0.5, 0.6, 0.4, 1.0])
        sigma = _model_implied_cov(A, S, spec.F)

        # Pack these values as theta
        spec.A_values[spec.A_free] = A[spec.A_free]
        spec.S_values[spec.S_free] = S[spec.S_free]
        theta = spec.pack_start()

        fml = ml_objective(theta, spec, sigma, 100)
        assert abs(fml) < 1e-8


class TestEstimate:
    def test_converges_on_simple_model(self):
        """Generate data from a known factor model and check convergence."""
        np.random.seed(42)
        n = 300

        # True model: f1 -> x1 (1.0), x2 (0.7), x3 (0.9)
        f1 = np.random.normal(0, 1, n)
        x1 = 1.0 * f1 + np.random.normal(0, 0.5, n)
        x2 = 0.7 * f1 + np.random.normal(0, 0.6, n)
        x3 = 0.9 * f1 + np.random.normal(0, 0.4, n)

        data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        tokens = parse_syntax("f1 =~ x1 + x2 + x3")
        spec = build_specification(tokens, data.columns.tolist())
        result = estimate(spec, data)

        assert result.converged

    def test_two_factor_model(self):
        """Two-factor CFA should converge."""
        np.random.seed(123)
        n = 500

        f1 = np.random.normal(0, 1, n)
        f2 = np.random.normal(0, 1, n) + 0.3 * f1

        x1 = 1.0 * f1 + np.random.normal(0, 0.5, n)
        x2 = 0.8 * f1 + np.random.normal(0, 0.5, n)
        x3 = 0.6 * f1 + np.random.normal(0, 0.5, n)
        x4 = 1.0 * f2 + np.random.normal(0, 0.5, n)
        x5 = 0.7 * f2 + np.random.normal(0, 0.5, n)
        x6 = 0.9 * f2 + np.random.normal(0, 0.5, n)

        data = pd.DataFrame({
            "x1": x1, "x2": x2, "x3": x3,
            "x4": x4, "x5": x5, "x6": x6,
        })

        model = """
            f1 =~ x1 + x2 + x3
            f2 =~ x4 + x5 + x6
        """
        tokens = parse_syntax(model)
        spec = build_specification(tokens, data.columns.tolist())
        result = estimate(spec, data)

        assert result.converged
        assert result.fmin < 1.0  # should have decent fit
