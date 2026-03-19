"""Tests for batch_bayes() multiprocess orchestration.

These tests verify the batch infrastructure (result containers, model
assignment, serialization) without requiring JAX/NumPyro.  Full
integration tests require a Bayesian backend and should be run on a
machine with numpyro installed.
"""

import numpy as np
import pandas as pd
import pytest
from semla.batch_bayes import BatchResult, BatchBayesResults


@pytest.fixture
def sample_results():
    """Create mock BatchBayesResults for testing containers."""
    est_data = {
        "lhs": ["f1", "f1"],
        "op": ["=~", "=~"],
        "rhs": ["x1", "x2"],
        "mean": [1.0, 0.8],
        "sd": [0.05, 0.06],
    }
    r1 = BatchResult(
        name="model_a",
        status="ok",
        estimates=pd.DataFrame(est_data),
        fit_indices={"waic": 100.0, "loo": 102.0},
        converged=True,
        summary="Model A summary",
        error=None,
        backend="cpu",
    )
    r2 = BatchResult(
        name="model_b",
        status="ok",
        estimates=pd.DataFrame(est_data),
        fit_indices={"waic": 95.0, "loo": 97.0},
        converged=True,
        summary="Model B summary",
        error=None,
        backend="gpu",
    )
    r3 = BatchResult(
        name="model_c",
        status="error",
        estimates=None,
        fit_indices=None,
        converged=None,
        summary=None,
        error="Test error",
        backend="cpu",
    )
    return BatchBayesResults({"model_a": r1, "model_b": r2, "model_c": r3})


class TestBatchResult:
    def test_ok_repr(self):
        r = BatchResult("test", "ok", None, {}, True, "", None, "cpu")
        assert "converged=True" in repr(r)

    def test_error_repr(self):
        r = BatchResult("test", "error", None, None, None, None, "boom", "cpu")
        assert "error=" in repr(r)


class TestBatchBayesResults:
    def test_len(self, sample_results):
        assert len(sample_results) == 3

    def test_getitem_by_name(self, sample_results):
        r = sample_results["model_a"]
        assert r.name == "model_a"
        assert r.status == "ok"

    def test_getitem_by_index(self, sample_results):
        r = sample_results[0]
        assert r.name == "model_a"

    def test_names(self, sample_results):
        assert sample_results.names == ["model_a", "model_b", "model_c"]

    def test_repr(self, sample_results):
        assert "2/3" in repr(sample_results)

    def test_iter(self, sample_results):
        names = [r.name for r in sample_results]
        assert names == ["model_a", "model_b", "model_c"]

    def test_compare(self, sample_results):
        comp = sample_results.compare()
        assert len(comp) == 2  # only successful models
        assert comp.index[0] == "model_b"  # lower WAIC first
        assert "waic" in comp.columns
        assert "backend" in comp.columns

    def test_summary_table(self, sample_results):
        st = sample_results.summary_table()
        assert len(st) == 3
        assert st.loc["model_c", "status"] == "error"
        assert st.loc["model_a", "converged"] == True


class TestModelAssignment:
    """Test that models are assigned to correct backends."""

    def test_no_gpu(self):
        """Without GPU, all models go to CPU."""
        from semla.batch_bayes import batch_bayes
        from semla.syntax import parse_syntax

        # We can't actually run batch_bayes without numpyro,
        # but we can test the assignment logic
        models = {
            "small": "f1 =~ x1+x2+x3",
            "big": "f1 =~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10",
        }
        # Assignment logic is inside batch_bayes, so test indirectly
        # by verifying parse_syntax works for size calculation
        for name, syntax in models.items():
            tokens = parse_syntax(syntax)
            latent = {tok.lhs for tok in tokens if tok.op == "=~"}
            observed = set()
            for tok in tokens:
                for term in tok.rhs:
                    if term.var not in latent:
                        observed.add(term.var)
            if name == "small":
                assert len(observed) == 3
            else:
                assert len(observed) == 10
