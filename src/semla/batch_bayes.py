"""Batch Bayesian SEM estimation with CPU+GPU parallelism.

Runs multiple Bayesian SEM models in parallel by distributing work across
CPU cores and (optionally) a GPU.  Each model runs in a separate process
to avoid JAX tracer conflicts.

Example::

    from semla import batch_bayes

    models = {
        "2factor": "f1 =~ x1+x2+x3\\nf2 =~ x4+x5+x6",
        "1factor": "f1 =~ x1+x2+x3+x4+x5+x6",
        "3factor": "f1 =~ x1+x2\\nf2 =~ x3+x4\\nf3 =~ x5+x6",
    }
    results = batch_bayes(models, data=df, cpu_cores=4)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _fit_worker(
    model_syntax: str,
    data_dict: dict,
    data_columns: list[str],
    backend: str,
    fit_kwargs: dict,
    func: str = "cfa",
) -> dict:
    """Fit a single Bayesian SEM model in a worker process.

    Returns a dict with serializable results (not the Model object, which
    cannot be pickled across processes).
    """
    # Force JAX backend before any JAX import
    if backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    # For GPU, let JAX auto-detect (default behavior)

    # Reconstruct DataFrame from dict
    data = pd.DataFrame(data_dict, columns=data_columns)

    from semla import cfa, sem, growth

    fit_fn = {"cfa": cfa, "sem": sem, "growth": growth}[func]

    fit_kwargs = dict(fit_kwargs)
    fit_kwargs["estimator"] = "bayes"
    fit_kwargs["progress_bar"] = False

    try:
        fit = fit_fn(model_syntax, data=data, **fit_kwargs)

        # Extract serializable results
        est = fit.estimates()
        fi = fit.fit_indices()

        return {
            "status": "ok",
            "estimates": est.to_dict("list"),
            "fit_indices": fi,
            "converged": fit.converged,
            "summary": fit.summary(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Results from a single model in a batch run."""
    name: str
    status: str
    estimates: pd.DataFrame | None
    fit_indices: dict | None
    converged: bool | None
    summary: str | None
    error: str | None
    backend: str

    def __repr__(self):
        if self.status == "ok":
            return f"BatchResult({self.name!r}, converged={self.converged}, backend={self.backend!r})"
        return f"BatchResult({self.name!r}, error={self.error!r})"


class BatchBayesResults:
    """Container for batch Bayesian estimation results.

    Supports indexing by name or position and provides a comparison table.
    """

    def __init__(self, results: dict[str, BatchResult]):
        self._results = results

    def __getitem__(self, key) -> BatchResult:
        if isinstance(key, int):
            return list(self._results.values())[key]
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results.values())

    def __repr__(self):
        ok = sum(1 for r in self._results.values() if r.status == "ok")
        return f"BatchBayesResults({ok}/{len(self)} models converged)"

    @property
    def names(self) -> list[str]:
        return list(self._results.keys())

    def compare(self) -> pd.DataFrame:
        """Compare all successfully fitted models by WAIC/fit indices."""
        rows = []
        for name, r in self._results.items():
            if r.status != "ok" or r.fit_indices is None:
                continue
            fi = r.fit_indices
            rows.append({
                "model": name,
                "waic": fi.get("waic", float("nan")),
                "loo": fi.get("loo", float("nan")),
                "backend": r.backend,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("waic").set_index("model")

    def summary_table(self) -> pd.DataFrame:
        """Overview table of all models."""
        rows = []
        for name, r in self._results.items():
            row = {"model": name, "status": r.status, "backend": r.backend}
            if r.status == "ok":
                row["converged"] = r.converged
                if r.fit_indices:
                    row["waic"] = r.fit_indices.get("waic", float("nan"))
            else:
                row["converged"] = None
                row["waic"] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """Check if a CUDA GPU is available without initializing JAX."""
    try:
        import jax
        return jax.default_backend() == "gpu"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def _count_model_complexity(syntax: str) -> int:
    """Count observed variables + free parameters as a complexity score."""
    from .syntax import parse_syntax

    tokens = parse_syntax(syntax)
    latent = {tok.lhs for tok in tokens if tok.op == "=~"}
    observed = set()
    n_free = 0
    for tok in tokens:
        for term in tok.rhs:
            if term.var not in latent:
                observed.add(term.var)
            if not term.fixed:
                n_free += 1
        if tok.op in ("~", "~~") and tok.lhs not in latent:
            observed.add(tok.lhs)
    # Complexity: observed vars (drives matrix size) + free params
    return len(observed) * 10 + n_free


def _collect_result(future, name, backend) -> BatchResult:
    """Collect a result from a completed future."""
    try:
        raw = future.result()
    except Exception as e:
        raw = {"status": "error", "error": str(e),
               "error_type": type(e).__name__}

    if raw["status"] == "ok":
        return BatchResult(
            name=name,
            status="ok",
            estimates=pd.DataFrame(raw["estimates"]),
            fit_indices=raw["fit_indices"],
            converged=raw["converged"],
            summary=raw["summary"],
            error=None,
            backend=backend,
        )
    return BatchResult(
        name=name,
        status="error",
        estimates=None,
        fit_indices=None,
        converged=None,
        summary=None,
        error=raw.get("error", "Unknown error"),
        backend=backend,
    )


def batch_bayes(
    models: dict[str, str],
    data: pd.DataFrame,
    *,
    func: str = "cfa",
    cpu_cores: int = 4,
    gpu: bool | str = "auto",
    warmup: int = 1000,
    draws: int = 1000,
    chains: int = 1,
    seed: int = 0,
    adapt_convergence: bool = True,
    positive_loadings: bool = True,
    priors: Any = None,
    adapt_delta: float = 0.8,
) -> BatchBayesResults:
    """Run multiple Bayesian SEM models in parallel.

    Uses a dual-queue scheduler: the GPU always works on the most complex
    remaining model, while CPU cores work on the least complex ones.  When
    a worker finishes, it picks up the next model from its end of the queue.

    Parameters
    ----------
    models : dict[str, str]
        Mapping of model names to lavaan-style syntax strings.
    data : pd.DataFrame
        Shared dataset for all models.
    func : str
        Fitting function: ``"cfa"`` (default), ``"sem"``, or ``"growth"``.
    cpu_cores : int
        Number of CPU worker processes (default 4).
    gpu : bool or "auto"
        Whether to use GPU.  ``"auto"`` detects availability.
    warmup : int
        Number of warmup samples per chain.
    draws : int
        Number of posterior draws per chain.
    chains : int
        Number of MCMC chains per model (default 1 for batch efficiency).
    seed : int
        Base random seed (each model gets seed + index).
    adapt_convergence : bool
        Whether to adaptively extend sampling for convergence.
    positive_loadings : bool
        Constrain first loading per factor to be positive.
    priors : str, dict, or None
        Prior specification (shared across all models).
    adapt_delta : float
        NUTS target acceptance probability.

    Returns
    -------
    BatchBayesResults
        Container with per-model results, comparison table, and summaries.
    """
    if len(models) == 0:
        raise ValueError("At least one model is required.")

    # Determine GPU availability
    use_gpu = False
    if gpu == "auto":
        use_gpu = _gpu_available()
    elif gpu:
        use_gpu = True

    # Prepare serializable data (dict of lists, not DataFrame)
    data_columns = list(data.columns)
    data_dict = {col: data[col].tolist() for col in data_columns}

    # Rank models by complexity
    complexity = {name: _count_model_complexity(syntax)
                  for name, syntax in models.items()}
    # Sorted most complex first
    ranked = sorted(complexity.items(), key=lambda x: -x[1])

    # Build the work queue: most complex → GPU, least complex → CPU
    # GPU takes from the front, CPU takes from the back
    work_queue = [name for name, _ in ranked]

    # Common fit kwargs
    fit_kwargs = {
        "warmup": warmup,
        "draws": draws,
        "chains": chains,
        "adapt_convergence": adapt_convergence,
        "positive_loadings": positive_loadings,
        "adapt_delta": adapt_delta,
    }
    if priors is not None:
        fit_kwargs["priors"] = priors

    # Assign seeds by original order
    model_seeds = {name: seed + i for i, name in enumerate(models)}

    total = len(models)
    print(f"batch_bayes: {total} models, "
          f"{cpu_cores} CPU workers"
          + (", 1 GPU worker" if use_gpu else "")
          + f" | ranked by complexity: "
          + ", ".join(f"{n}({complexity[n]})" for n, _ in ranked))

    ctx = mp.get_context("spawn")
    results = {}
    active_futures = {}  # future -> (name, backend)

    with ProcessPoolExecutor(max_workers=cpu_cores + (1 if use_gpu else 0),
                             mp_context=ctx) as pool:

        gpu_busy = False

        def _submit(name, backend):
            kw = dict(fit_kwargs)
            kw["seed"] = model_seeds[name]
            future = pool.submit(
                _fit_worker, models[name], data_dict, data_columns,
                backend, kw, func,
            )
            active_futures[future] = (name, backend)

        def _fill_slots():
            """Fill available CPU and GPU slots from the work queue."""
            nonlocal gpu_busy

            # GPU: take the most complex remaining model (front of queue)
            if use_gpu and not gpu_busy and work_queue:
                name = work_queue.pop(0)
                _submit(name, "gpu")
                gpu_busy = True

            # CPU: take the least complex remaining models (back of queue)
            cpu_active = sum(1 for _, (_, b) in active_futures.items()
                           if b == "cpu")
            while cpu_active < cpu_cores and work_queue:
                name = work_queue.pop()  # pop from back (least complex)
                _submit(name, "cpu")
                cpu_active += 1

        # Initial fill
        _fill_slots()

        # Process completions and refill
        while active_futures:
            # Wait for any future to complete
            done = set()
            for f in list(active_futures):
                if f.done():
                    done.add(f)

            if not done:
                # Brief wait, then check again
                import time
                time.sleep(0.1)
                continue

            for future in done:
                name, backend = active_futures.pop(future)
                result = _collect_result(future, name, backend)
                results[name] = result
                print(f"  {name}: {result.status} ({backend})")

                if backend == "gpu":
                    gpu_busy = False

            # Refill slots with remaining work
            _fill_slots()

    # Return in original order
    ordered = {name: results[name] for name in models if name in results}
    return BatchBayesResults(ordered)
