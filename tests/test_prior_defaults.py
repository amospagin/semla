"""Tests for semla.prior_defaults — adaptive/weak prior resolution."""

import numpy as np
import pytest

from semla.priors import Normal, HalfCauchy, InverseGamma, Gamma, Prior
from semla.prior_defaults import (
    resolve_priors,
    _param_key,
    _matrix_category,
    MATRIX_KEYS,
)
from semla.specification import build_specification, ParamInfo
from semla.syntax import parse_syntax


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfa_spec_and_data():
    """Build a simple 2-factor CFA spec with synthetic data."""
    syntax = """
        f1 =~ x1 + x2 + x3
        f2 =~ x4 + x5 + x6
    """
    tokens = parse_syntax(syntax)
    observed = ["x1", "x2", "x3", "x4", "x5", "x6"]
    spec = build_specification(tokens, observed)

    rng = np.random.default_rng(42)
    # data with different SDs per column
    data = rng.normal(loc=0, scale=[1, 2, 3, 4, 5, 6], size=(200, 6))
    return spec, data


@pytest.fixture(scope="module")
def regression_spec_and_data():
    """Build a SEM spec with regressions."""
    syntax = """
        f1 =~ x1 + x2 + x3
        f2 =~ x4 + x5 + x6
        f2 ~ f1
    """
    tokens = parse_syntax(syntax)
    observed = ["x1", "x2", "x3", "x4", "x5", "x6"]
    spec = build_specification(tokens, observed, auto_cov_latent=False)

    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=2, size=(200, 6))
    return spec, data


# ── param_key / matrix_category ────────────────────────────────────────

class TestParamKey:
    def test_loading_key(self):
        p = ParamInfo(lhs="f1", op="=~", rhs="x2", free=True, value=0.5)
        assert _param_key(p) == "f1=~x2"

    def test_variance_key(self):
        p = ParamInfo(lhs="x1", op="~~", rhs="x1", free=True, value=0.5)
        assert _param_key(p) == "x1~~x1"

    def test_regression_key(self):
        p = ParamInfo(lhs="f2", op="~", rhs="f1", free=True, value=0.0)
        assert _param_key(p) == "f2~f1"


class TestMatrixCategory:
    def test_loading(self, cfa_spec_and_data):
        spec, _ = cfa_spec_and_data
        p = ParamInfo(lhs="f1", op="=~", rhs="x2", free=True, value=0.5)
        assert _matrix_category(p, spec) == "loadings"

    def test_residual_variance(self, cfa_spec_and_data):
        spec, _ = cfa_spec_and_data
        p = ParamInfo(lhs="x1", op="~~", rhs="x1", free=True, value=0.5)
        assert _matrix_category(p, spec) == "residual_variances"

    def test_factor_variance(self, cfa_spec_and_data):
        spec, _ = cfa_spec_and_data
        p = ParamInfo(lhs="f1", op="~~", rhs="f1", free=True, value=0.5)
        assert _matrix_category(p, spec) == "factor_variances"

    def test_covariance(self, cfa_spec_and_data):
        spec, _ = cfa_spec_and_data
        p = ParamInfo(lhs="f1", op="~~", rhs="f2", free=True, value=0.05)
        assert _matrix_category(p, spec) == "covariances"

    def test_regression(self, regression_spec_and_data):
        spec, _ = regression_spec_and_data
        p = ParamInfo(lhs="f2", op="~", rhs="f1", free=True, value=0.0)
        assert _matrix_category(p, spec) == "regressions"


# ── adaptive priors (default) ──────────────────────────────────────────

class TestAdaptivePriors:
    def test_returns_prior_for_every_free_param(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        n_free = sum(1 for p in spec.params if p.free)
        assert len(result) == n_free

    def test_all_values_are_prior_instances(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        for key, prior in result.items():
            assert isinstance(prior, Prior), f"{key} is not a Prior"

    def test_loading_scales_by_indicator_sd(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        # x2 has SD ~2, x5 has SD ~5
        p_x2 = result.get("f1=~x2")
        p_x5 = result.get("f2=~x5")
        assert isinstance(p_x2, Normal)
        assert isinstance(p_x5, Normal)
        # x5 prior should be wider than x2
        assert p_x5.sigma > p_x2.sigma

    def test_residual_variance_is_inverse_gamma(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        p = result.get("x1~~x1")
        assert isinstance(p, InverseGamma)

    def test_factor_variance_is_inverse_gamma(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        p = result.get("f1~~f1")
        assert isinstance(p, InverseGamma)

    def test_covariance_prior_is_normal(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        p = result.get("f1~~f2")
        assert isinstance(p, Normal)

    def test_regression_prior(self, regression_spec_and_data):
        spec, data = regression_spec_and_data
        result = resolve_priors(spec, data)
        p = result.get("f2~f1")
        assert isinstance(p, Normal)


# ── weak priors ─────────────────────────────────────────────────────────

class TestWeakPriors:
    def test_returns_prior_for_every_free_param(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data, priors="weak")
        n_free = sum(1 for p in spec.params if p.free)
        assert len(result) == n_free

    def test_loading_is_wide_normal(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data, priors="weak")
        p = result.get("f1=~x2")
        assert isinstance(p, Normal)
        assert p.sigma == 5.0

    def test_variance_is_inverse_gamma(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data, priors="weak")
        p = result.get("x1~~x1")
        assert isinstance(p, InverseGamma)
        assert p.concentration == 2.0

    def test_unknown_preset_raises(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        with pytest.raises(ValueError, match="Unknown prior preset"):
            resolve_priors(spec, data, priors="strong")


# ── matrix-level overrides ──────────────────────────────────────────────

class TestMatrixOverrides:
    def test_loading_override(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        custom = Normal(mu=0.0, sigma=0.5)
        result = resolve_priors(spec, data, priors={"loadings": custom})
        # all loadings should use the override
        for key, prior in result.items():
            if "=~" in key:
                assert prior is custom

    def test_variance_override_leaves_loadings_adaptive(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        custom_var = InverseGamma(concentration=5.0, rate=5.0)
        result = resolve_priors(
            spec, data,
            priors={"residual_variances": custom_var},
        )
        # loadings should still be adaptive Normal
        for key, prior in result.items():
            if "=~" in key:
                assert isinstance(prior, Normal)
                assert prior.sigma != 10.0  # not weak default


# ── per-parameter overrides ─────────────────────────────────────────────

class TestPerParamOverrides:
    def test_single_param_override(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        custom = Normal(mu=0.7, sigma=0.2)
        result = resolve_priors(spec, data, priors={"f1=~x2": custom})
        assert result["f1=~x2"] is custom
        # other loadings should be adaptive
        assert result.get("f1=~x3") is not custom

    def test_param_overrides_matrix_level(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        matrix_prior = Normal(mu=0.0, sigma=0.5)
        param_prior = Normal(mu=0.7, sigma=0.2)
        result = resolve_priors(
            spec, data,
            priors={"loadings": matrix_prior, "f1=~x2": param_prior},
        )
        # f1=~x2 should use param override, not matrix override
        assert result["f1=~x2"] is param_prior
        # f1=~x3 should use matrix override
        assert result["f1=~x3"] is matrix_prior

    def test_unused_override_warns(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        with pytest.warns(UserWarning, match="unknown parameters ignored"):
            resolve_priors(spec, data, priors={"f99=~z1": Normal(0, 1)})


# ── validation ──────────────────────────────────────────────────────────

class TestValidation:
    def test_non_prior_value_raises(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        with pytest.raises(TypeError, match="must be a Prior instance"):
            resolve_priors(spec, data, priors={"loadings": 5.0})

    def test_fixed_params_excluded(self, cfa_spec_and_data):
        spec, data = cfa_spec_and_data
        result = resolve_priors(spec, data)
        # first loading per factor is fixed to 1.0 — should not appear
        assert "f1=~x1" not in result
        assert "f2=~x4" not in result

    def test_matrix_keys_constant(self):
        assert MATRIX_KEYS == {
            "loadings", "regressions", "residual_variances",
            "factor_variances", "covariances", "intercepts",
        }
