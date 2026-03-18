"""Tests for reliability measures and factor scores."""

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


class TestReliability:
    def test_returns_dict(self, hs_fit):
        rel = hs_fit.reliability()
        assert isinstance(rel, dict)

    def test_all_factors_present(self, hs_fit):
        rel = hs_fit.reliability()
        assert set(rel.keys()) == {"visual", "textual", "speed"}

    def test_omega_between_0_and_1(self, hs_fit):
        for factor, vals in hs_fit.reliability().items():
            assert 0 < vals["omega"] < 1, f"omega for {factor} out of range"

    def test_alpha_between_0_and_1(self, hs_fit):
        for factor, vals in hs_fit.reliability().items():
            assert 0 < vals["alpha"] < 1, f"alpha for {factor} out of range"

    def test_textual_highest_reliability(self, hs_fit):
        """Textual factor is known to have highest reliability."""
        rel = hs_fit.reliability()
        assert rel["textual"]["omega"] > rel["visual"]["omega"]
        assert rel["textual"]["omega"] > rel["speed"]["omega"]

    def test_omega_geq_alpha(self, hs_fit):
        """Omega should be >= alpha (omega is less conservative)."""
        for factor, vals in hs_fit.reliability().items():
            assert vals["omega"] >= vals["alpha"] - 0.01, (
                f"{factor}: omega={vals['omega']:.3f} < alpha={vals['alpha']:.3f}"
            )


class TestFactorScores:
    def test_returns_dataframe(self, hs_fit):
        scores = hs_fit.predict()
        assert isinstance(scores, pd.DataFrame)

    def test_correct_shape(self, hs_fit):
        scores = hs_fit.predict()
        assert scores.shape == (301, 3)

    def test_correct_columns(self, hs_fit):
        scores = hs_fit.predict()
        assert list(scores.columns) == ["visual", "textual", "speed"]

    def test_mean_near_zero(self, hs_fit):
        scores = hs_fit.predict()
        for col in scores.columns:
            assert abs(scores[col].mean()) < 0.01

    def test_regression_method(self, hs_fit):
        scores = hs_fit.predict(method="regression")
        assert scores.shape[1] == 3

    def test_bartlett_method(self, hs_fit):
        scores = hs_fit.predict(method="bartlett")
        assert scores.shape[1] == 3

    def test_regression_bartlett_correlated(self, hs_fit):
        """Both methods should give highly correlated scores."""
        reg = hs_fit.predict(method="regression")
        bart = hs_fit.predict(method="bartlett")
        for col in reg.columns:
            r = reg[col].corr(bart[col])
            assert r > 0.95, f"{col}: correlation between methods = {r:.3f}"

    def test_invalid_method_raises(self, hs_fit):
        with pytest.raises(ValueError, match="method must be"):
            hs_fit.predict(method="invalid")

    def test_predict_on_new_data(self, hs_fit):
        """Should work on a subset of the data."""
        df = HolzingerSwineford1939().head(50)
        scores = hs_fit.predict(data=df)
        assert scores.shape == (50, 3)
