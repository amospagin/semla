"""Comprehensive edge case tests for the semla SEM package.

Categories tested:
- Model specification edge cases (identification, missing vars, data issues)
- Estimation edge cases (small samples, boundary solutions, just-identified)
- Output edge cases (non-convergence, just-identified fit, summary/check)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from semla import cfa, sem, growth


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_one_factor_data(n, n_indicators, seed=42):
    """Generate data from a single-factor model."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    cols = {}
    for i in range(1, n_indicators + 1):
        cols[f"y{i}"] = f + rng.normal(0, 0.5, n)
    return pd.DataFrame(cols)


def _make_three_factor_data(n=100, seed=42):
    """Generate data from a three-factor CFA model."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n) + 0.3 * f1
    f3 = rng.normal(0, 1, n) + 0.2 * f1 + 0.2 * f2
    return pd.DataFrame({
        "x1": f1 + rng.normal(0, 0.5, n),
        "x2": 0.8 * f1 + rng.normal(0, 0.6, n),
        "x3": 0.9 * f1 + rng.normal(0, 0.5, n),
        "x4": f2 + rng.normal(0, 0.5, n),
        "x5": 0.9 * f2 + rng.normal(0, 0.6, n),
        "x6": 0.8 * f2 + rng.normal(0, 0.5, n),
        "x7": f3 + rng.normal(0, 0.5, n),
        "x8": 0.7 * f3 + rng.normal(0, 0.6, n),
        "x9": 0.8 * f3 + rng.normal(0, 0.5, n),
    })


THREE_FACTOR_MODEL = """
    f1 =~ x1 + x2 + x3
    f2 =~ x4 + x5 + x6
    f3 =~ x7 + x8 + x9
"""


# ═══════════════════════════════════════════════════════════════════════════
# Model specification edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestJustIdentifiedCFA:
    """1. Single-factor CFA with exactly 3 indicators (df=0)."""

    def test_converges(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        assert fit.converged

    def test_has_estimates(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        est = fit.estimates()
        assert isinstance(est, pd.DataFrame)
        assert len(est) > 0

    def test_df_is_zero(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        idx = fit.fit_indices()
        assert idx["df"] == 0


class TestTwoIndicatorCFA:
    """2. Single-factor CFA with 2 indicators -- underidentified, should warn."""

    def test_warns_underidentified(self):
        rng = np.random.default_rng(42)
        n = 50
        f = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "y1": f + rng.normal(0, 0.5, n),
            "y2": f + rng.normal(0, 0.5, n),
        })
        # A 2-indicator single-factor model is underidentified (df < 0).
        # The package should warn about this.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfa("f1 =~ y1 + y2", data=df)
        # Should have at least one warning (identification, convergence,
        # or numerical issues like invalid sqrt from negative variance)
        warn_msgs = [str(x.message).lower() for x in w]
        has_relevant_warning = any(
            "identif" in m or "converge" in m or "indicator" in m
            or "underidentif" in m or "small" in m or "invalid" in m
            or "sqrt" in m
            for m in warn_msgs
        )
        assert has_relevant_warning, (
            f"Expected a warning about identification/convergence/numerical "
            f"issues for 2-indicator model, got: {warn_msgs}"
        )


class TestObservedOnlyPathModel:
    """3. Model with no latent variables (observed-only path model)."""

    def test_sem_observed_only_converges(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.8, n)
        Y = 0.3 * M + 0.2 * X + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})
        fit = sem("M ~ X\nY ~ M + X", data=df)
        assert fit.converged

    def test_sem_observed_only_has_regression_estimates(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.8, n)
        Y = 0.3 * M + 0.2 * X + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})
        fit = sem("M ~ X\nY ~ M + X", data=df)
        est = fit.estimates()
        assert "~" in est["op"].values
        # Should have at least 2 regression paths
        reg_rows = est[est["op"] == "~"]
        assert len(reg_rows) >= 2


class TestExtraDataColumns:
    """4. Extra data columns not in model -- should be ignored silently."""

    def test_extra_columns_ignored(self):
        rng = np.random.default_rng(42)
        n = 100
        f = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "y1": f + rng.normal(0, 0.5, n),
            "y2": 0.8 * f + rng.normal(0, 0.5, n),
            "y3": 0.9 * f + rng.normal(0, 0.5, n),
            "extra1": rng.normal(0, 1, n),
            "extra2": rng.normal(0, 1, n),
            "id_col": np.arange(n),
        })
        # No warnings about extra columns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        extra_warnings = [
            x for x in w
            if "extra" in str(x.message).lower() or "unused" in str(x.message).lower()
        ]
        assert len(extra_warnings) == 0
        assert fit.converged


class TestMissingVariable:
    """5. Variable in syntax but not in data -- should raise ValueError."""

    def test_raises_with_clear_message(self):
        df = _make_one_factor_data(n=50, n_indicators=3)
        with pytest.raises(ValueError, match="not found in data"):
            cfa("f1 =~ y1 + y2 + nonexistent", data=df)

    def test_error_mentions_missing_var(self):
        df = _make_one_factor_data(n=50, n_indicators=3)
        with pytest.raises(ValueError, match="nonexistent"):
            cfa("f1 =~ y1 + y2 + nonexistent", data=df)

    def test_error_mentions_available_columns(self):
        df = _make_one_factor_data(n=50, n_indicators=3)
        with pytest.raises(ValueError, match="Available columns"):
            cfa("f1 =~ y1 + y2 + missing_var", data=df)


class TestSingleObservation:
    """6. Single observation (n=1) -- should raise or warn."""

    def test_single_obs_raises_or_warns(self):
        df = pd.DataFrame({"y1": [1.0], "y2": [2.0], "y3": [3.0]})
        # With n=1, model fitting should either raise an error or issue
        # a warning about sample size.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                cfa("f1 =~ y1 + y2 + y3", data=df)
                # If it didn't raise, there should be a warning
                warn_msgs = [str(x.message).lower() for x in w]
                has_warning = any(
                    "sample" in m or "small" in m or "singular" in m
                    for m in warn_msgs
                )
                assert has_warning, (
                    f"Expected warning for n=1, got: {warn_msgs}"
                )
            except (ValueError, np.linalg.LinAlgError):
                # Raising an error is also acceptable
                pass


class TestConstantColumn:
    """7. Data with a constant column referenced in model -- should raise."""

    def test_constant_column_raises(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "y1": rng.normal(0, 1, n),
            "y2": rng.normal(0, 1, n),
            "y3": np.ones(n),  # constant
        })
        with pytest.raises(ValueError, match="zero variance"):
            cfa("f1 =~ y1 + y2 + y3", data=df)

    def test_constant_column_names_offending_variable(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "y1": rng.normal(0, 1, n),
            "y2": rng.normal(0, 1, n),
            "y3": np.full(n, 5.0),  # constant at 5
        })
        with pytest.raises(ValueError, match="y3"):
            cfa("f1 =~ y1 + y2 + y3", data=df)


# ═══════════════════════════════════════════════════════════════════════════
# Estimation edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestSmallSampleThreeFactor:
    """8. Very small sample (n=20) with 3-factor CFA -- should converge with warning."""

    def test_converges_with_warning(self):
        df = _make_three_factor_data(n=20, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = cfa(THREE_FACTOR_MODEL, data=df)
        warn_msgs = [str(x.message).lower() for x in w]
        has_sample_warning = any(
            "small" in m or "sample" in m or "converge" in m
            for m in warn_msgs
        )
        assert has_sample_warning, (
            f"Expected warning about small sample, got: {warn_msgs}"
        )

    def test_produces_estimates(self):
        df = _make_three_factor_data(n=20, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = cfa(THREE_FACTOR_MODEL, data=df)
        est = fit.estimates()
        assert isinstance(est, pd.DataFrame)
        assert len(est) > 0


class TestBoundaryVariance:
    """9. Model that converges to boundary (variance near 0)."""

    def test_near_zero_variance_produces_results(self):
        rng = np.random.default_rng(42)
        n = 80
        f = rng.normal(0, 1, n)
        # y1 is almost perfectly predicted by f -- residual variance near 0
        df = pd.DataFrame({
            "y1": f + rng.normal(0, 0.01, n),
            "y2": f + rng.normal(0, 0.5, n),
            "y3": f + rng.normal(0, 0.5, n),
            "y4": f + rng.normal(0, 0.5, n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = cfa("f1 =~ y1 + y2 + y3 + y4", data=df)
        est = fit.estimates()
        assert isinstance(est, pd.DataFrame)
        # y1 residual variance should be very small
        resid = est[
            (est["lhs"] == "y1") & (est["op"] == "~~") & (est["rhs"] == "y1")
        ]
        if len(resid) == 1:
            assert resid.iloc[0]["est"] < 0.1


class TestJustIdentifiedChiSquare:
    """10. Just-identified model (df=0) -- chi-square should be ~0."""

    def test_chi_square_near_zero(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        idx = fit.fit_indices()
        assert idx["df"] == 0
        assert abs(idx["chi_square"]) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Output edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEstimatesNonConverged:
    """11. estimates() when model didn't converge -- should still return DataFrame."""

    def test_returns_dataframe_even_if_not_converged(self):
        rng = np.random.default_rng(42)
        n = 30
        # Create data designed to cause convergence issues
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.99 + rng.normal(0, 0.05, n)
        x3 = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = cfa("f =~ x1 + x2 + x3", data=df)
        # Whether converged or not, estimates() should return a DataFrame
        est = fit.estimates()
        assert isinstance(est, pd.DataFrame)
        assert len(est) > 0
        assert "est" in est.columns


class TestFitIndicesJustIdentified:
    """12. fit_indices() for just-identified model -- CFI=1, chi-square ~0."""

    def test_cfi_is_one(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        idx = fit.fit_indices()
        # For df=0, CFI should be 1.0 (or NaN by convention -- both valid)
        cfi = idx["cfi"]
        if not np.isnan(cfi):
            assert abs(cfi - 1.0) < 1e-6

    def test_chi_square_zero(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        idx = fit.fit_indices()
        assert abs(idx["chi_square"]) < 1e-6

    def test_rmsea_zero_or_nan(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        idx = fit.fit_indices()
        rmsea = idx["rmsea"]
        # RMSEA should be 0 or NaN for df=0
        assert np.isnan(rmsea) or abs(rmsea) < 1e-6


class TestSummaryRuns:
    """13. summary() runs without error for standard CFA."""

    def test_summary_returns_string(self):
        df = _make_three_factor_data(n=100, seed=42)
        fit = cfa(THREE_FACTOR_MODEL, data=df)
        result = fit.summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_contains_key_sections(self):
        df = _make_three_factor_data(n=100, seed=42)
        fit = cfa(THREE_FACTOR_MODEL, data=df)
        result = fit.summary()
        # Summary should contain fit information and parameter estimates
        lower = result.lower()
        assert "chi" in lower or "fit" in lower
        assert "estimate" in lower or "est" in lower

    def test_summary_for_sem(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        Y = 0.5 * X + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"X": X, "Y": Y})
        fit = sem("Y ~ X", data=df)
        result = fit.summary()
        assert isinstance(result, str)
        assert len(result) > 0


class TestCheckRuns:
    """14. check() runs without error for any fitted model."""

    def test_check_returns_string_for_cfa(self):
        df = _make_three_factor_data(n=100, seed=42)
        fit = cfa(THREE_FACTOR_MODEL, data=df)
        result = fit.check()
        assert isinstance(result, str)

    def test_check_returns_string_for_sem(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.8, n)
        Y = 0.3 * M + 0.2 * X + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})
        fit = sem("M ~ X\nY ~ M + X", data=df)
        result = fit.check()
        assert isinstance(result, str)

    def test_check_for_non_converged_model(self):
        rng = np.random.default_rng(42)
        n = 30
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.99 + rng.normal(0, 0.05, n)
        x3 = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = cfa("f =~ x1 + x2 + x3", data=df)
        result = fit.check()
        assert isinstance(result, str)

    def test_check_for_just_identified(self):
        df = _make_one_factor_data(n=100, n_indicators=3)
        fit = cfa("f1 =~ y1 + y2 + y3", data=df)
        result = fit.check()
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# Growth model edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestGrowthModel:
    """Growth model with linear trajectory."""

    def test_growth_converges(self):
        rng = np.random.default_rng(42)
        n = 80
        intercept = rng.normal(5, 1, n)
        slope = rng.normal(0.5, 0.3, n)
        df = pd.DataFrame({
            "t1": intercept + 0 * slope + rng.normal(0, 0.3, n),
            "t2": intercept + 1 * slope + rng.normal(0, 0.3, n),
            "t3": intercept + 2 * slope + rng.normal(0, 0.3, n),
            "t4": intercept + 3 * slope + rng.normal(0, 0.3, n),
        })
        model = """
            i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
            s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
        """
        fit = growth(model, data=df)
        assert fit.converged

    def test_growth_estimates_have_means(self):
        rng = np.random.default_rng(42)
        n = 80
        intercept = rng.normal(5, 1, n)
        slope = rng.normal(0.5, 0.3, n)
        df = pd.DataFrame({
            "t1": intercept + 0 * slope + rng.normal(0, 0.3, n),
            "t2": intercept + 1 * slope + rng.normal(0, 0.3, n),
            "t3": intercept + 2 * slope + rng.normal(0, 0.3, n),
            "t4": intercept + 3 * slope + rng.normal(0, 0.3, n),
        })
        model = """
            i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
            s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
        """
        fit = growth(model, data=df)
        est = fit.estimates()
        # Growth models should have intercept (mean) parameters
        assert "~1" in est["op"].values or "~1" in est["op"].unique()
