"""NumPyro model builder for Bayesian SEM estimation.

Translates a :class:`~semla.specification.ModelSpecification` and a resolved
prior dict into a NumPyro probabilistic model that can be sampled with NUTS.

Architecture::

    ModelSpecification + priors ──► _build_numpyro_model() ──► numpyro model fn
                                                             ──► NUTS sampler
                                                             ──► posterior draws
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .prior_defaults import resolve_priors, _param_key, _matrix_category, PriorSpec
from .specification import ModelSpecification, ParamInfo


def _import_jax():
    try:
        import jax.numpy as jnp
        return jnp
    except ImportError:
        raise ImportError(
            "jax is required for Bayesian estimation. "
            "Install it with:  pip install semla[bayes]"
        ) from None


def _import_numpyro():
    try:
        import numpyro
        return numpyro
    except ImportError:
        raise ImportError(
            "numpyro is required for Bayesian estimation. "
            "Install it with:  pip install semla[bayes]"
        ) from None


# ---------------------------------------------------------------------------
# Build the parameter mapping tables (numpy, computed once before sampling)
# ---------------------------------------------------------------------------

def _build_param_table(spec: ModelSpecification):
    """Pre-compute the mapping from free parameters to RAM matrix positions.

    Returns
    -------
    a_params : list of (key, row, col, raw_idx, eff_idx)
        Free A-matrix parameters.
    s_params : list of (key, row, col, raw_idx, eff_idx)
        Free S-matrix parameters (lower triangle).
    m_params : list of (key, var_idx, raw_idx, eff_idx)
        Free mean-structure parameters.
    n_effective : int
        Number of unique free parameters after equality constraints.
    param_keys : list[str]
        Ordered list of effective parameter keys (length n_effective).
    """
    a_params = []
    s_params = []
    m_params = []

    # We need to walk free params in the same order as pack_start / unpack
    # Order: A-matrix (row-major), then S lower triangle (row-major), then m
    raw_idx = 0

    # A-matrix free params (row-major order)
    n = spec.n_vars
    a_free_flat = spec.A_free.ravel()
    a_positions = []
    for flat in range(n * n):
        if a_free_flat[flat]:
            r, c = divmod(flat, n)
            a_positions.append((r, c, raw_idx))
            raw_idx += 1

    # S-matrix free params (lower triangle, row-major)
    s_lower = spec._S_free_lower
    s_positions = []
    for r in range(n):
        for c in range(r + 1):
            if s_lower[r, c]:
                s_positions.append((r, c, raw_idx))
                raw_idx += 1

    # m free params
    m_positions = []
    if spec.meanstructure and spec.m_free is not None:
        for i in range(n):
            if spec.m_free[i]:
                m_positions.append((i, raw_idx))
                raw_idx += 1

    # Map raw_idx -> effective_idx
    cmap = spec._constraint_map

    # Build a raw_idx -> param key mapping by matching against spec.params
    # We need param keys for the prior dict lookup
    # Match A params to ParamInfo
    for r, c, ridx in a_positions:
        eidx = int(cmap[ridx]) if cmap is not None else ridx
        # find corresponding ParamInfo
        for p in spec.params:
            if not p.free:
                continue
            if p.op == "=~" and spec._idx(p.rhs) == r and spec._idx(p.lhs) == c:
                a_params.append((_param_key(p), r, c, ridx, eidx))
                break
            elif p.op == "~" and spec._idx(p.lhs) == r and spec._idx(p.rhs) == c:
                a_params.append((_param_key(p), r, c, ridx, eidx))
                break

    for r, c, ridx in s_positions:
        eidx = int(cmap[ridx]) if cmap is not None else ridx
        for p in spec.params:
            if not p.free:
                continue
            if p.op == "~~":
                pi = spec._idx(p.lhs)
                pj = spec._idx(p.rhs)
                pr, pc = max(pi, pj), min(pi, pj)
                if pr == r and pc == c:
                    s_params.append((_param_key(p), r, c, ridx, eidx))
                    break

    for i, ridx in m_positions:
        eidx = int(cmap[ridx]) if cmap is not None else ridx
        for p in spec.params:
            if not p.free:
                continue
            if p.op == "~1" and spec._idx(p.lhs) == i:
                m_params.append((_param_key(p), i, ridx, eidx))
                break

    n_effective = spec.n_free
    # Build ordered list of effective param keys
    eff_key_map: dict[int, str] = {}
    for key, _, _, _, eidx in a_params:
        if eidx not in eff_key_map:
            eff_key_map[eidx] = key
    for key, _, _, _, eidx in s_params:
        if eidx not in eff_key_map:
            eff_key_map[eidx] = key
    for key, _, _, eidx in m_params:
        if eidx not in eff_key_map:
            eff_key_map[eidx] = key
    param_keys = [eff_key_map[i] for i in range(n_effective)]

    return a_params, s_params, m_params, n_effective, param_keys


# ---------------------------------------------------------------------------
# NumPyro model function
# ---------------------------------------------------------------------------

def build_numpyro_model(
    spec: ModelSpecification,
    data: np.ndarray,
    priors: PriorSpec = None,
):
    """Build a NumPyro model function from a RAM specification.

    Parameters
    ----------
    spec : ModelSpecification
        The built RAM specification.
    data : np.ndarray
        Observed data matrix (N x p), columns matching ``spec.observed_vars``.
    priors : str, dict, or None
        Prior specification (see :func:`~semla.prior_defaults.resolve_priors`).

    Returns
    -------
    model_fn : callable
        A NumPyro model function suitable for use with ``numpyro.infer.MCMC``.
    prior_dict : dict[str, Prior]
        The resolved prior for every free parameter.
    param_keys : list[str]
        Ordered list of effective parameter names.
    """
    jnp = _import_jax()
    numpyro = _import_numpyro()

    prior_dict = resolve_priors(spec, data, priors)
    a_params, s_params, m_params, n_effective, param_keys = _build_param_table(spec)

    # Pre-convert fixed matrices to JAX arrays
    A_fixed = jnp.array(spec.A_values)
    S_fixed = jnp.array(spec.S_values)
    F = jnp.array(spec.F)
    I = jnp.eye(spec.n_vars)
    obs_data = jnp.array(data)

    m_fixed = None
    if spec.meanstructure and spec.m_values is not None:
        m_fixed = jnp.array(spec.m_values)

    # Group params by effective index (for equality constraints)
    # eff_idx -> list of (matrix, row, col) placements
    eff_a_placements: dict[int, list[tuple[int, int]]] = {}
    for key, r, c, ridx, eidx in a_params:
        eff_a_placements.setdefault(eidx, []).append((r, c))

    eff_s_placements: dict[int, list[tuple[int, int]]] = {}
    for key, r, c, ridx, eidx in s_params:
        eff_s_placements.setdefault(eidx, []).append((r, c))

    eff_m_placements: dict[int, list[int]] = {}
    for key, i, ridx, eidx in m_params:
        eff_m_placements.setdefault(eidx, []).append(i)

    # Determine which S params are variances (diagonal) for positivity
    variance_keys = set()
    for p in spec.params:
        if p.free and p.op == "~~" and p.lhs == p.rhs:
            variance_keys.add(_param_key(p))

    def model():
        # Sample each effective parameter from its prior
        theta = {}
        for eidx in range(n_effective):
            key = param_keys[eidx]
            prior = prior_dict[key]
            theta[eidx] = numpyro.sample(key, prior.to_numpyro())

        # Build A matrix
        A = A_fixed.copy()
        for eidx, placements in eff_a_placements.items():
            val = theta[eidx]
            for r, c in placements:
                A = A.at[r, c].set(val)

        # Build S matrix (symmetric)
        S = S_fixed.copy()
        for eidx, placements in eff_s_placements.items():
            val = theta[eidx]
            for r, c in placements:
                S = S.at[r, c].set(val)
                if r != c:
                    S = S.at[c, r].set(val)

        # Model-implied covariance: Sigma = F @ (I-A)^{-1} @ S @ ((I-A)^{-1})^T @ F^T
        I_A_inv = jnp.linalg.inv(I - A)
        Sigma = F @ I_A_inv @ S @ I_A_inv.T @ F.T

        # Regularise for numerical stability
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + 1e-6 * jnp.eye(spec.n_obs)

        # Model-implied mean (if meanstructure)
        if m_fixed is not None and m_params:
            m = m_fixed.copy()
            for eidx, placements in eff_m_placements.items():
                val = theta[eidx]
                for i in placements:
                    m = m.at[i].set(val)
            mu = (F @ I_A_inv @ m)
        else:
            mu = jnp.zeros(spec.n_obs)

        # Likelihood
        import numpyro.distributions as dist
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(loc=mu, covariance_matrix=Sigma),
            obs=obs_data,
        )

    return model, prior_dict, param_keys


# ---------------------------------------------------------------------------
# Convenience: run MCMC
# ---------------------------------------------------------------------------

def run_mcmc(
    spec: ModelSpecification,
    data: np.ndarray,
    priors: PriorSpec = None,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 0,
    target_accept_prob: float = 0.8,
    progress_bar: bool = True,
) -> "MCMCResult":
    """Run NUTS sampling on a Bayesian SEM model.

    Parameters
    ----------
    spec : ModelSpecification
        Built RAM specification.
    data : np.ndarray
        Observed data (N x p).
    priors : str, dict, or None
        Prior specification.
    num_warmup, num_samples, num_chains : int
        MCMC settings.
    seed : int
        Random seed.
    target_accept_prob : float
        NUTS target acceptance probability.
    progress_bar : bool
        Show progress bar during sampling.

    Returns
    -------
    MCMCResult
        Container with posterior samples and diagnostics.
    """
    import jax
    jnp = _import_jax()
    numpyro = _import_numpyro()
    from numpyro.infer import MCMC, NUTS

    model_fn, prior_dict, param_keys = build_numpyro_model(spec, data, priors)

    kernel = NUTS(model_fn, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key)

    return MCMCResult(
        mcmc=mcmc,
        spec=spec,
        param_keys=param_keys,
        prior_dict=prior_dict,
    )


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

class MCMCResult:
    """Container for Bayesian SEM MCMC results.

    Attributes
    ----------
    mcmc : numpyro.infer.MCMC
        The underlying MCMC object.
    samples : dict[str, jax.Array]
        Posterior samples keyed by parameter name.
    param_keys : list[str]
        Ordered parameter names.
    """

    def __init__(self, mcmc, spec, param_keys, prior_dict):
        self.mcmc = mcmc
        self.spec = spec
        self.param_keys = param_keys
        self.prior_dict = prior_dict
        self.samples = mcmc.get_samples()

    def summary(self) -> "pd.DataFrame":
        """Return a summary DataFrame with posterior mean, SD, and quantiles."""
        import pandas as pd
        jnp = _import_jax()
        np_ = np

        rows = []
        for key in self.param_keys:
            if key not in self.samples:
                continue
            s = np.array(self.samples[key])
            rows.append({
                "parameter": key,
                "mean": float(np_.mean(s)),
                "sd": float(np_.std(s)),
                "q025": float(np_.percentile(s, 2.5)),
                "q25": float(np_.percentile(s, 25)),
                "q50": float(np_.percentile(s, 50)),
                "q75": float(np_.percentile(s, 75)),
                "q975": float(np_.percentile(s, 97.5)),
                "n_eff": float(_effective_sample_size(s)),
                "rhat": float(_rhat(s, self.mcmc.num_chains)),
            })
        return pd.DataFrame(rows)

    def print_summary(self):
        """Print MCMC diagnostics via numpyro."""
        self.mcmc.print_summary()


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _effective_sample_size(x: np.ndarray) -> float:
    """Estimate effective sample size using autocorrelation."""
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return float(n)

    # Compute autocorrelation via FFT
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (var * n)

    # Geyer's initial positive sequence estimator
    total = 0.0
    for i in range(1, n, 2):
        pair_sum = acf[i] + (acf[i + 1] if i + 1 < n else 0.0)
        if pair_sum < 0:
            break
        total += pair_sum
    tau = 1.0 + 2.0 * total
    return n / max(tau, 1.0)


def _rhat(x: np.ndarray, num_chains: int) -> float:
    """Compute split R-hat for a flat array of samples."""
    if num_chains <= 1:
        return float("nan")
    n_total = len(x)
    n_per = n_total // num_chains
    if n_per < 2:
        return float("nan")
    chains = x[:n_per * num_chains].reshape(num_chains, n_per)

    # Split each chain in half
    half = n_per // 2
    splits = np.vstack([chains[:, :half], chains[:, half:2 * half]])
    m = splits.shape[0]
    n = splits.shape[1]

    chain_means = splits.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean(splits.var(axis=1, ddof=1))

    if W == 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))
