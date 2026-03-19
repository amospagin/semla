"""Robust ML (MLR) standard errors and scaled chi-square.

Implements the Huber-White sandwich estimator and Satorra-Bentler
scaled test statistic for non-normal continuous data.
"""

from __future__ import annotations

import numpy as np

from .estimation import _model_implied_cov, _compute_se
from .specification import ModelSpecification


def _vech_indices(p: int) -> list[tuple[int, int]]:
    """Return (i, j) pairs for vech ordering (lower triangle incl. diagonal)."""
    return [(i, j) for i in range(p) for j in range(i + 1)]


def compute_gamma(data: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
    """Compute the Gamma matrix (asymptotic covariance of vech(S)).

    Gamma_ab = (1/N) sum_k (x_ki*x_kj - s_ij)(x_km*x_kn - s_mn)

    where a=(i,j) and b=(m,n) are vech indices.

    Parameters
    ----------
    data : np.ndarray
        Centered data matrix (n x p).
    sample_cov : np.ndarray
        Sample covariance matrix (p x p).

    Returns
    -------
    np.ndarray
        Gamma matrix, shape (p*(p+1)/2, p*(p+1)/2).
    """
    n, p = data.shape
    indices = _vech_indices(p)
    m = len(indices)

    # Compute outer products minus expected for each observation
    # w_k[a] = x_ki * x_kj - s_ij for vech index a=(i,j)
    W = np.zeros((n, m))
    for a, (i, j) in enumerate(indices):
        W[:, a] = data[:, i] * data[:, j] - sample_cov[i, j]

    # Gamma = W'W / n — ML-consistent estimate
    Gamma = (W.T @ W) / n
    return Gamma


def compute_robust_se(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    gamma: np.ndarray,
    raw_data: np.ndarray = None,
) -> np.ndarray:
    """Compute robust (sandwich) standard errors for ML.

    Uses the Gamma-based sandwich formula:
        V = (Δ'WΔ)⁻¹ @ (Δ'WΓW Δ) @ (Δ'WΔ)⁻¹
    where Δ = Jacobian of vech(Σ) w.r.t. θ, W = ML weight matrix,
    and Γ = asymptotic covariance of vech(S).

    SE = sqrt(diag(V) / N)
    """
    eps = 1e-7
    k = len(theta)

    A_mat, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A_mat, S_mat, spec.F)
    if sigma is None:
        return np.full(k, np.nan)

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    p = sigma.shape[0]
    indices = _vech_indices(p)
    m = len(indices)

    # Jacobian Δ: d(vech(Σ))/d(θ), shape (m, k)
    Delta = np.zeros((m, k))
    for i in range(k):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        A_p, S_p = spec.unpack(theta_plus)
        A_m, S_m = spec.unpack(theta_minus)
        sig_p = _model_implied_cov(A_p, S_p, spec.F)
        sig_m = _model_implied_cov(A_m, S_m, spec.F)
        if sig_p is None or sig_m is None:
            return np.full(k, np.nan)
        dSig = (sig_p - sig_m) / (2 * eps)
        for a, (r, c) in enumerate(indices):
            Delta[a, i] = dSig[r, c]

    # Weight matrix W_ml in vech form
    W_ml = np.zeros((m, m))
    for a, (i, j) in enumerate(indices):
        for b, (r, c) in enumerate(indices):
            val = sigma_inv[i, r] * sigma_inv[j, c]
            if r != c:
                val += sigma_inv[i, c] * sigma_inv[j, r]
            W_ml[a, b] = val

    # Bread: J = Δ'WΔ
    WDelta = W_ml @ Delta  # (m, k)
    J = Delta.T @ WDelta   # (k, k)
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    # Meat: Δ'W Γ W Δ
    meat = WDelta.T @ gamma @ WDelta  # (k, k)

    # Sandwich: V = J⁻¹ meat J⁻¹, SE = sqrt(diag(V) / N)
    V = J_inv @ meat @ J_inv

    var_theta = np.diag(V) / n_obs
    se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
    return se


def satorra_bentler_chi_square(
    fmin: float,
    n_obs: int,
    df: int,
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    gamma: np.ndarray,
) -> tuple[float, float]:
    """Compute Satorra-Bentler scaled chi-square.

    T_scaled = T_ML / c where c = tr(UG) / df

    Returns (T_scaled, scaling_factor).
    """
    T_ml = (n_obs - 1) * fmin
    if df <= 0:
        return T_ml, 1.0

    eps = 1e-7
    k = len(theta)
    A, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return T_ml, 1.0

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return T_ml, 1.0

    p = sigma.shape[0]
    indices = _vech_indices(p)
    m = len(indices)

    # Jacobian Delta (m x k)
    Delta = np.zeros((m, k))
    for i in range(k):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        A_p, S_p = spec.unpack(theta_plus)
        A_m, S_m = spec.unpack(theta_minus)
        sig_p = _model_implied_cov(A_p, S_p, spec.F)
        sig_m = _model_implied_cov(A_m, S_m, spec.F)
        if sig_p is None or sig_m is None:
            return T_ml, 1.0
        dSig = (sig_p - sig_m) / (2 * eps)
        for a, (r, c) in enumerate(indices):
            Delta[a, i] = dSig[r, c]

    # W_ml (vech version of Sigma_inv kron Sigma_inv)
    W_ml = np.zeros((m, m))
    for a, (i, j) in enumerate(indices):
        for b, (r, c) in enumerate(indices):
            val = sigma_inv[i, r] * sigma_inv[j, c]
            if r != c:
                val += sigma_inv[i, c] * sigma_inv[j, r]
            W_ml[a, b] = val

    # UG = W_ml - W_ml Delta (Delta' W_ml Delta)^{-1} Delta' W_ml
    try:
        DtW = Delta.T @ W_ml
        DtWD_inv = np.linalg.inv(DtW @ Delta)
        UG = W_ml - W_ml @ Delta @ DtWD_inv @ DtW
    except np.linalg.LinAlgError:
        return T_ml, 1.0

    # Scaling factor
    trace_UG_Gamma = np.trace(UG @ gamma)
    if trace_UG_Gamma > 0:
        c = trace_UG_Gamma / df
        return T_ml / c, c
    else:
        return T_ml, 1.0
