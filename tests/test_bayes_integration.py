"""Integration test: wire estimator='bayes' into cfa()/sem() (#32)."""

import numpy as np
import pandas as pd
import pytest

numpyro = pytest.importorskip("numpyro")

from semla import cfa, sem
from semla.bayes_results import BayesianResults


@pytest.fixture(scope="module")
def cfa_data():
    """Synthetic CFA data as a DataFrame."""
    rng = np.random.default_rng(42)
    n = 200
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    return pd.DataFrame({
        "x1": 1.0 * f1 + rng.normal(0, 0.5, n),
        "x2": 0.8 * f1 + rng.normal(0, 0.5, n),
        "x3": 0.6 * f1 + rng.normal(0, 0.5, n),
        "x4": 1.0 * f2 + rng.normal(0, 0.5, n),
        "x5": 0.9 * f2 + rng.normal(0, 0.5, n),
        "x6": 0.7 * f2 + rng.normal(0, 0.5, n),
    })


class TestCfaBayes:
    @pytest.fixture(scope="class")
    def fit(self, cfa_data):
        return cfa(
            "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
            data=cfa_data,
            estimator="bayes",
            warmup=30,
            draws=30,
            chains=2,
            seed=0,
            adapt_convergence=False,
            progress_bar=False,
        )

    def test_returns_model(self, fit):
        from semla.model import Model
        assert isinstance(fit, Model)

    def test_results_are_bayesian(self, fit):
        assert isinstance(fit.results, BayesianResults)

    def test_summary_works(self, fit):
        s = fit.summary()
        assert "Bayesian" in s

    def test_estimates_works(self, fit):
        est = fit.estimates()
        assert len(est) > 0
        assert "mean" in est.columns

    def test_fit_indices_works(self, fit):
        fi = fit.fit_indices()
        assert "waic" in fi

    def test_converged_attr(self, fit):
        assert isinstance(fit.converged, bool)


class TestSemBayes:
    def test_sem_bayes_runs(self, cfa_data):
        fit = sem(
            "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf2 ~ f1",
            data=cfa_data,
            estimator="bayes",
            warmup=30,
            draws=30,
            chains=2,
            seed=0,
            adapt_convergence=False,
            progress_bar=False,
        )
        assert isinstance(fit.results, BayesianResults)


class TestBayesWithPriors:
    def test_weak_priors(self, cfa_data):
        fit = cfa(
            "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
            data=cfa_data,
            estimator="bayes",
            priors="weak",
            warmup=30,
            draws=30,
            chains=2,
            seed=0,
            adapt_convergence=False,
            progress_bar=False,
        )
        assert isinstance(fit.results, BayesianResults)

    def test_custom_priors(self, cfa_data):
        from semla.priors import Normal
        fit = cfa(
            "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6",
            data=cfa_data,
            estimator="bayes",
            priors={"loadings": Normal(0, 5)},
            warmup=30,
            draws=30,
            chains=2,
            seed=0,
            adapt_convergence=False,
            progress_bar=False,
        )
        assert isinstance(fit.results, BayesianResults)


class TestBayesErrorHandling:
    def test_unknown_estimator(self, cfa_data):
        with pytest.raises(ValueError, match="Unknown estimator"):
            cfa(
                "f1 =~ x1 + x2 + x3",
                data=cfa_data,
                estimator="unknown",
            )
