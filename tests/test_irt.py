"""Tests for IRT models."""

import numpy as np
import pandas as pd
import pytest
from semla import irt


@pytest.fixture(scope="module")
def binary_data():
    """Generate binary IRT data from known 2PL model."""
    rng = np.random.default_rng(42)
    n = 500
    theta = rng.normal(0, 1, n)
    true_a = [1.0, 1.5, 0.8, 1.2, 0.6]
    true_b = [-1.0, -0.5, 0.0, 0.5, 1.0]
    data = {}
    for i in range(5):
        p = 1 / (1 + np.exp(-1.7 * true_a[i] * (theta - true_b[i])))
        data[f"item{i+1}"] = (rng.random(n) < p).astype(float)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def items():
    return [f"item{i+1}" for i in range(5)]


@pytest.fixture(scope="module")
def fit_2pl(items, binary_data):
    return irt(items, data=binary_data, model_type="2PL")


class TestIRTBasic:
    def test_converges(self, fit_2pl):
        assert fit_2pl.converged

    def test_irt_params_dataframe(self, fit_2pl):
        params = fit_2pl.irt_params()
        assert isinstance(params, pd.DataFrame)
        assert "discrimination" in params.columns
        assert "difficulty" in params.columns

    def test_all_items_present(self, fit_2pl, items):
        params = fit_2pl.irt_params()
        assert set(params["item"]) == set(items)

    def test_discriminations_positive(self, fit_2pl):
        params = fit_2pl.irt_params()
        assert (params["discrimination"] > 0).all()

    def test_difficulties_ordered(self, fit_2pl):
        """Items were generated with increasing difficulty."""
        params = fit_2pl.irt_params()
        diffs = params.sort_values("item")["difficulty"].values
        # Not strictly ordered due to estimation noise, but trend should hold
        assert diffs[-1] > diffs[0]


class TestIRTMethods:
    def test_icc_shape(self, fit_2pl):
        icc = fit_2pl.icc()
        assert "theta" in icc.columns
        assert icc.shape[0] == 61  # default theta grid
        assert icc.shape[1] == 6  # theta + 5 items

    def test_icc_bounded(self, fit_2pl):
        """ICC probabilities should be between 0 and 1."""
        icc = fit_2pl.icc()
        for col in icc.columns:
            if col != "theta":
                assert (icc[col] >= 0).all()
                assert (icc[col] <= 1).all()

    def test_item_information(self, fit_2pl):
        info = fit_2pl.item_information()
        assert "theta" in info.columns
        # Information should be non-negative
        for col in info.columns:
            if col != "theta":
                assert (info[col] >= 0).all()

    def test_test_information(self, fit_2pl):
        ti = fit_2pl.test_information()
        assert "information" in ti.columns
        assert "se" in ti.columns
        assert (ti["information"] > 0).all()
        assert (ti["se"] > 0).all()

    def test_abilities(self, fit_2pl):
        abilities = fit_2pl.abilities()
        assert "theta" in abilities.columns
        assert len(abilities) == 500
        # Mean should be approximately 0
        assert abs(abilities["theta"].mean()) < 0.2

    def test_summary(self, fit_2pl):
        output = fit_2pl.summary()
        assert "2PL" in output
        assert "Discrim" in output


class TestIRTModelTypes:
    def test_1pl(self, items, binary_data):
        fit = irt(items, data=binary_data, model_type="1PL")
        assert fit.converged
        assert fit.model_type == "1PL"

    def test_grm_with_ordinal(self):
        """GRM with ordinal (5-category) data."""
        rng = np.random.default_rng(42)
        n = 300
        theta = rng.normal(0, 1, n)
        data = {}
        for i in range(4):
            z = theta + rng.normal(0, 0.5, n)
            data[f"item{i+1}"] = pd.cut(z, bins=5, labels=[1, 2, 3, 4, 5]).astype(float)
        df = pd.DataFrame(data)
        fit = irt([f"item{i+1}" for i in range(4)], data=df, model_type="GRM")
        assert fit.converged

    def test_invalid_model_type(self, items, binary_data):
        with pytest.raises(ValueError, match="model_type must be"):
            irt(items, data=binary_data, model_type="3PL")
