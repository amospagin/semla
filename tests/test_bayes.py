"""Tests for semla.bayes — NumPyro model builder for Bayesian SEM."""

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import jax
import jax.numpy as jnp

from semla.specification import build_specification, ParamInfo
from semla.syntax import parse_syntax
from semla.priors import Normal, InverseGamma
from semla.bayes import (
    build_numpyro_model,
    _build_param_table,
    run_mcmc,
    MCMCResult,
    _effective_sample_size,
    _rhat,
)


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_cfa():
    """Simple 2-factor CFA with synthetic data from known parameters."""
    rng = np.random.default_rng(42)
    n = 200

    # True factor scores
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)

    # True loadings and residuals
    x1 = 1.0 * f1 + rng.normal(0, 0.5, n)
    x2 = 0.8 * f1 + rng.normal(0, 0.5, n)
    x3 = 0.6 * f1 + rng.normal(0, 0.5, n)
    x4 = 1.0 * f2 + rng.normal(0, 0.5, n)
    x5 = 0.9 * f2 + rng.normal(0, 0.5, n)
    x6 = 0.7 * f2 + rng.normal(0, 0.5, n)

    data = np.column_stack([x1, x2, x3, x4, x5, x6])

    syntax = """
        f1 =~ x1 + x2 + x3
        f2 =~ x4 + x5 + x6
    """
    tokens = parse_syntax(syntax)
    observed = ["x1", "x2", "x3", "x4", "x5", "x6"]
    spec = build_specification(tokens, observed)

    return spec, data


# ── param table tests ───────────────────────────────────────────────────

class TestParamTable:
    def test_correct_number_of_effective_params(self, simple_cfa):
        spec, data = simple_cfa
        a_params, s_params, m_params, n_eff, keys = _build_param_table(spec)
        assert n_eff == spec.n_free

    def test_param_keys_length(self, simple_cfa):
        spec, data = simple_cfa
        _, _, _, n_eff, keys = _build_param_table(spec)
        assert len(keys) == n_eff

    def test_a_params_are_loadings(self, simple_cfa):
        spec, data = simple_cfa
        a_params, _, _, _, _ = _build_param_table(spec)
        # Should have 4 free loadings (2 per factor minus fixed first)
        assert len(a_params) == 4
        for key, r, c, ridx, eidx in a_params:
            assert "=~" in key

    def test_s_params_include_variances_and_covariance(self, simple_cfa):
        spec, data = simple_cfa
        _, s_params, _, _, _ = _build_param_table(spec)
        # 6 residual vars + 2 factor vars + 1 factor covariance = 9
        assert len(s_params) == 9


# ── model build tests ──────────────────────────────────────────────────

class TestBuildModel:
    def test_returns_callable(self, simple_cfa):
        spec, data = simple_cfa
        model_fn, prior_dict, param_keys = build_numpyro_model(spec, data)
        assert callable(model_fn)

    def test_prior_dict_covers_all_free(self, simple_cfa):
        spec, data = simple_cfa
        _, prior_dict, param_keys = build_numpyro_model(spec, data)
        n_free = sum(1 for p in spec.params if p.free)
        assert len(prior_dict) == n_free

    def test_param_keys_match_prior_dict(self, simple_cfa):
        spec, data = simple_cfa
        _, prior_dict, param_keys = build_numpyro_model(spec, data)
        for key in param_keys:
            assert key in prior_dict

    def test_model_traces_without_error(self, simple_cfa):
        """Model function runs under numpyro.handlers.trace."""
        spec, data = simple_cfa
        model_fn, _, _ = build_numpyro_model(spec, data)
        import numpyro.handlers as handlers
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()
        assert "obs" in trace

    def test_custom_priors(self, simple_cfa):
        spec, data = simple_cfa
        model_fn, prior_dict, _ = build_numpyro_model(
            spec, data, priors={"loadings": Normal(0, 5)}
        )
        for key, prior in prior_dict.items():
            if "=~" in key:
                assert isinstance(prior, Normal)
                assert prior.sigma == 5


# ── mean structure ──────────────────────────────────────────────────────

class TestMeanStructure:
    def test_model_with_meanstructure(self):
        syntax = """
            f1 =~ x1 + x2 + x3
        """
        tokens = parse_syntax(syntax)
        observed = ["x1", "x2", "x3"]
        spec = build_specification(tokens, observed, meanstructure=True)

        rng = np.random.default_rng(0)
        data = rng.normal(5, 2, (100, 3))

        model_fn, prior_dict, param_keys = build_numpyro_model(spec, data)

        # Should include intercept parameters
        intercept_keys = [k for k in param_keys if "~1" in k]
        assert len(intercept_keys) == 3

        # Trace should work
        import numpyro.handlers as handlers
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()
        assert "obs" in trace


# ── equality constraints ────────────────────────────────────────────────

class TestEqualityConstraints:
    def test_constrained_params_share_sample(self):
        syntax = """
            f1 =~ x1 + c1*x2 + c1*x3
        """
        tokens = parse_syntax(syntax)
        observed = ["x1", "x2", "x3"]
        spec = build_specification(tokens, observed)

        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (100, 3))

        model_fn, prior_dict, param_keys = build_numpyro_model(spec, data)

        # c1 constraint means x2 and x3 loadings share a parameter
        # So n_effective should be less than n_raw_free
        loading_keys = [k for k in param_keys if "=~" in k]
        assert len(loading_keys) == 1  # only one unique loading (c1)


# ── diagnostics helpers ─────────────────────────────────────────────────

class TestDiagnostics:
    def test_ess_iid_sample(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        ess = _effective_sample_size(x)
        # For iid samples, ESS should be close to n
        assert ess > 500

    def test_ess_constant(self):
        x = np.ones(100)
        ess = _effective_sample_size(x)
        assert ess == 100.0

    def test_rhat_single_chain(self):
        x = np.random.randn(100)
        r = _rhat(x, 1)
        assert np.isnan(r)

    def test_rhat_converged(self):
        rng = np.random.default_rng(42)
        # 4 chains of iid draws from same distribution
        x = rng.normal(0, 1, 4000)
        r = _rhat(x, 4)
        assert 0.9 < r < 1.1


# ── MCMC smoke test (short run) ────────────────────────────────────────

class TestMCMCSmoke:
    @pytest.mark.slow
    def test_short_mcmc_runs(self, simple_cfa):
        """Very short MCMC run to verify the pipeline works end-to-end."""
        spec, data = simple_cfa
        result = run_mcmc(
            spec, data,
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
            progress_bar=False,
        )
        assert isinstance(result, MCMCResult)
        assert len(result.samples) > 0

        df = result.summary()
        assert "mean" in df.columns
        assert "rhat" in df.columns
        assert len(df) == spec.n_free
