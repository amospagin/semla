"""Tests for fit.check() post-estimation diagnostics."""

import numpy as np
import pandas as pd
import pytest
import warnings
from semla import cfa, sem
from semla.datasets import HolzingerSwineford1939


@pytest.fixture(scope="module")
def hs_data():
    return HolzingerSwineford1939()


class TestCheckGoodModel:
    """A well-fitting model should report minimal issues."""

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa("""
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """, data=hs_data)

    def test_returns_string(self, fit):
        result = fit.check()
        assert isinstance(result, str)

    def test_no_convergence_issue(self, fit):
        result = fit.check()
        assert "NON-CONVERGENCE" not in result

    def test_no_heywood(self, fit):
        result = fit.check()
        assert "HEYWOOD" not in result

    def test_no_boundary(self, fit):
        result = fit.check()
        assert "BOUNDARY" not in result

    def test_no_identification_issue(self, fit):
        result = fit.check()
        assert "IDENTIFICATION" not in result


class TestCheckPoorModel:
    """A poorly fitting model should report misfit."""

    @pytest.fixture(scope="class")
    def fit(self, hs_data):
        return cfa("g =~ x1+x2+x3+x4+x5+x6+x7+x8+x9", data=hs_data)

    def test_reports_misfit(self, fit):
        result = fit.check()
        assert "MISFIT" in result

    def test_reports_large_residuals(self, fit):
        result = fit.check()
        assert "standardized residual" in result

    def test_suggests_modindices(self, fit):
        result = fit.check()
        assert "modindices" in result


class TestCheckHeywood:
    """A model with Heywood cases should report them."""

    @pytest.fixture(scope="class")
    def fit(self):
        rng = np.random.default_rng(42)
        n = 50
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.99 + rng.normal(0, 0.1, n)
        x3 = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cfa("f =~ x1 + x2 + x3", data=df)

    def test_reports_heywood(self, fit):
        result = fit.check()
        assert "HEYWOOD" in result

    def test_reports_issue(self, fit):
        """Heywood model should report at least one issue."""
        result = fit.check()
        assert "issue(s) detected" in result


class TestConvergenceWarning:
    """Non-convergence warnings should include diagnostics."""

    def test_warning_has_gradient_norm(self):
        rng = np.random.default_rng(42)
        n = 15
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.01, n)  # near-perfect collinearity
        x3 = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfa("f =~ x1 + x2 + x3", data=df)

        # Should get either convergence or heywood warning
        relevant = [x for x in w if "converge" in str(x.message).lower()
                    or "heywood" in str(x.message).lower()]
        assert len(relevant) >= 1
        # If convergence warning, check it has diagnostics
        conv_warnings = [x for x in w if "converge" in str(x.message).lower()]
        if conv_warnings:
            msg = str(conv_warnings[0].message)
            assert "gradient norm" in msg
            assert "Possible causes" in msg

    def test_heywood_warning_has_suggestions(self):
        rng = np.random.default_rng(42)
        n = 50
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.99 + rng.normal(0, 0.1, n)
        x3 = rng.normal(0, 1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfa("f =~ x1 + x2 + x3", data=df)

        heywood_warnings = [x for x in w if "heywood" in str(x.message).lower()]
        assert len(heywood_warnings) >= 1
        msg = str(heywood_warnings[0].message)
        assert "Too few indicators" in msg
        assert "modindices" in msg


class TestCheckMediationModel:
    """A just-identified mediation model should report no issues."""

    @pytest.fixture(scope="class")
    def fit(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, n)
        M = 0.5 * X + rng.normal(0, 0.8, n)
        Y = 0.4 * M + 0.2 * X + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})
        return sem("M ~ X; Y ~ M + X", data=df)

    def test_no_issues(self, fit):
        result = fit.check()
        assert "NON-CONVERGENCE" not in result
        assert "HEYWOOD" not in result
        assert "BOUNDARY" not in result
