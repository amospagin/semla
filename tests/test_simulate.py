"""Tests for Monte Carlo power analysis (simulate_power)."""

import numpy as np
import pytest

from semla.simulate import simulate_power


# Simple 1-factor, 3-indicator CFA model
MODEL = "f =~ x1 + x2 + x3"

# Population with strong loadings (0.8) — first loading fixed to 1.0
POPULATION_STRONG = {
    ("f", "=~", "x2"): 0.8,
    ("f", "=~", "x3"): 0.8,
    ("x1", "~~", "x1"): 0.36,
    ("x2", "~~", "x2"): 0.36,
    ("x3", "~~", "x3"): 0.36,
    ("f", "~~", "f"): 1.0,
}

# Population with zero loading on x3 (for Type I error test)
POPULATION_ZERO_EFFECT = {
    ("f", "=~", "x2"): 0.8,
    ("f", "=~", "x3"): 0.0,   # null effect
    ("x1", "~~", "x1"): 0.36,
    ("x2", "~~", "x2"): 0.36,
    ("x3", "~~", "x3"): 1.0,  # full variance since loading is 0
    ("f", "~~", "f"): 1.0,
}


N_REPS = 100  # for speed in testing


class TestPowerHighEffect:
    """Power for large effects should be near 1.0."""

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_power(
            MODEL, POPULATION_STRONG, n=300, n_replications=N_REPS, seed=42
        )

    def test_power_near_one(self, result):
        pw = result.power()
        # Free loadings (x2, x3) have pop value 0.8 — should have high power
        loading_rows = pw[pw["op"] == "=~"]
        for _, row in loading_rows.iterrows():
            assert row["power"] > 0.85, (
                f"Power for {row['lhs']} {row['op']} {row['rhs']} = {row['power']:.3f}, "
                f"expected > 0.85"
            )

    def test_convergence_rate(self, result):
        assert result.convergence_rate > 0.9, (
            f"Convergence rate = {result.convergence_rate:.3f}, expected > 0.9"
        )


class TestTypeIError:
    """Power for a zero effect should be near alpha (Type I error rate)."""

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_power(
            MODEL, POPULATION_ZERO_EFFECT, n=300, n_replications=N_REPS, seed=123
        )

    def test_type_i_error_near_alpha(self, result):
        pw = result.power()
        # Find the x3 loading (pop value = 0)
        x3_row = pw[(pw["op"] == "=~") & (pw["rhs"] == "x3")]
        assert len(x3_row) == 1
        power_x3 = x3_row.iloc[0]["power"]
        # Type I error should be near 0.05, allow generous tolerance for 100 reps
        assert power_x3 < 0.20, (
            f"Type I error rate = {power_x3:.3f}, expected < 0.20"
        )


class TestBias:
    """Average bias should be near zero for converged replications."""

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_power(
            MODEL, POPULATION_STRONG, n=300, n_replications=N_REPS, seed=42
        )

    def test_bias_near_zero(self, result):
        bi = result.bias()
        for _, row in bi.iterrows():
            # Bias should be small relative to the parameter value
            abs_bias = abs(row["bias"])
            assert abs_bias < 0.10, (
                f"Bias for {row['lhs']} {row['op']} {row['rhs']} = {row['bias']:.4f}, "
                f"expected |bias| < 0.10"
            )


class TestCoverage:
    """95% CI coverage rate should be near 0.95."""

    @pytest.fixture(scope="class")
    def result(self):
        return simulate_power(
            MODEL, POPULATION_STRONG, n=300, n_replications=N_REPS, seed=42
        )

    def test_coverage_near_95(self, result):
        cv = result.coverage()
        for _, row in cv.iterrows():
            # With 100 reps, allow coverage between 0.80 and 1.0
            assert 0.80 <= row["coverage"] <= 1.0, (
                f"Coverage for {row['lhs']} {row['op']} {row['rhs']} = "
                f"{row['coverage']:.3f}, expected between 0.80 and 1.0"
            )


class TestConvergence:
    """Convergence rate should be > 0.9 for well-specified models."""

    def test_convergence_rate(self):
        result = simulate_power(
            MODEL, POPULATION_STRONG, n=300, n_replications=N_REPS, seed=42
        )
        assert result.convergence_rate > 0.9, (
            f"Convergence rate = {result.convergence_rate:.3f}, expected > 0.9"
        )


class TestSummary:
    """Test that summary() returns a non-empty string."""

    def test_summary_string(self):
        result = simulate_power(
            MODEL, POPULATION_STRONG, n=200, n_replications=20, seed=42
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "Monte Carlo Power Analysis" in s
        assert "Replications" in s

    def test_repr(self):
        result = simulate_power(
            MODEL, POPULATION_STRONG, n=200, n_replications=20, seed=42
        )
        assert "Monte Carlo" in repr(result)
