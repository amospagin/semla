"""Maximum Likelihood estimation for SEM using RAM notation.

Model-implied covariance:
    Sigma(theta) = F @ inv(I - A) @ S @ inv(I - A)^T @ F^T

ML fitting function:
    F_ML = log|Sigma| + tr(S_sample @ Sigma^{-1}) - log|S_sample| - p
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from .specification import ModelSpecification


@dataclass
class EstimationResult:
    """Raw output from the optimizer."""

    converged: bool
    iterations: int
    fmin: float  # minimum of F_ML
    theta: np.ndarray  # optimal parameter vector
    hessian_inv: np.ndarray | None  # approximate inverse Hessian
    sample_cov: np.ndarray
    n_obs: int
    spec: ModelSpecification


def _model_implied_cov(
    A: np.ndarray, S: np.ndarray, F: np.ndarray
) -> np.ndarray:
    """Compute model-implied covariance matrix using RAM."""
    n = A.shape[0]
    I = np.eye(n)
    try:
        IminA_inv = np.linalg.inv(I - A)
    except np.linalg.LinAlgError:
        return None
    implied_full = IminA_inv @ S @ IminA_inv.T
    return F @ implied_full @ F.T


def ml_objective(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
) -> float:
    """Compute the ML discrepancy function.

    Parameters
    ----------
    theta : array
        Free parameter vector.
    spec : ModelSpecification
        Model specification with matrix templates.
    sample_cov : array
        Sample covariance matrix (p x p).
    n_obs : int
        Number of observations.

    Returns
    -------
    float
        F_ML value (to be minimized).
    """
    A, S_mat = spec.unpack(theta)
    p = sample_cov.shape[0]

    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return 1e10

    try:
        sign, logdet_sigma = np.linalg.slogdet(sigma)
        if sign <= 0:
            return 1e10

        sigma_inv = np.linalg.inv(sigma)
        sign_s, logdet_s = np.linalg.slogdet(sample_cov)

        fml = logdet_sigma + np.trace(sample_cov @ sigma_inv) - logdet_s - p
        return fml

    except np.linalg.LinAlgError:
        return 1e10


def ml_gradient(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Compute the gradient of F_ML via finite differences."""
    eps = 1e-7
    grad = np.zeros_like(theta)
    f0 = ml_objective(theta, spec, sample_cov, n_obs)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (ml_objective(theta_plus, spec, sample_cov, n_obs) - f0) / eps
    return grad


def _compute_se(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Compute standard errors using the expected information matrix.

    Uses the analytical formula for expected information under ML:
        I_ij = (N-1)/2 * tr(Sigma^{-1} dSigma_i Sigma^{-1} dSigma_j)
        Var(theta) = inv(I)

    First derivatives of Sigma w.r.t. theta are computed numerically.
    This is more stable than computing second derivatives of F_ML.
    """
    eps = 1e-7
    k = len(theta)

    # Model-implied covariance at the optimum
    A, S_mat = spec.unpack(theta)
    sigma = _model_implied_cov(A, S_mat, spec.F)
    if sigma is None:
        return np.full(k, np.nan)

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)

    # Compute dSigma/dtheta_i numerically (central differences)
    p = sigma.shape[0]
    dSigma = np.zeros((k, p, p))
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

        dSigma[i] = (sig_p - sig_m) / (2 * eps)

    # Expected information matrix:
    # I_ij = (N-1)/2 * tr(Sigma^{-1} @ dSigma_i @ Sigma^{-1} @ dSigma_j)
    # Precompute Sigma^{-1} @ dSigma_i for each i
    SinvdS = np.zeros((k, p, p))
    for i in range(k):
        SinvdS[i] = sigma_inv @ dSigma[i]

    info = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            info[i, j] = 0.5 * (n_obs - 1) * np.trace(SinvdS[i] @ SinvdS[j])
            info[j, i] = info[i, j]

    # Var(theta) = inv(I), SE = sqrt(diag(inv(I)))
    try:
        info_inv = np.linalg.inv(info)
        var_theta = np.diag(info_inv)
        se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
        return se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def estimate(
    spec: ModelSpecification,
    data: pd.DataFrame,
) -> EstimationResult:
    """Estimate model parameters via Maximum Likelihood.

    Parameters
    ----------
    spec : ModelSpecification
        Model specification from ``build_specification()``.
    data : pd.DataFrame
        Data with columns matching observed variables.

    Returns
    -------
    EstimationResult
    """
    # Compute sample covariance matrix
    obs_data = data[spec.observed_vars].values
    n_obs = obs_data.shape[0]
    sample_cov = np.cov(obs_data, rowvar=False, ddof=1)

    # Make sure it's 2D even with 1 variable
    if sample_cov.ndim == 0:
        sample_cov = sample_cov.reshape(1, 1)

    # Starting values
    theta0 = spec.pack_start()

    # Optimize with BFGS
    result = optimize.minimize(
        ml_objective,
        theta0,
        args=(spec, sample_cov, n_obs),
        method="BFGS",
        jac=ml_gradient,
        options={"maxiter": 10000, "gtol": 1e-6},
    )

    # Polish: re-run from the BFGS solution with tighter tolerance
    if result.success:
        result2 = optimize.minimize(
            ml_objective,
            result.x,
            args=(spec, sample_cov, n_obs),
            method="BFGS",
            jac=ml_gradient,
            options={"maxiter": 10000, "gtol": 1e-9},
        )
        if result2.fun <= result.fun + 1e-10:
            result2.success = True  # accept if objective didn't increase
            result = result2

    if not result.success:
        warnings.warn(
            f"Optimization did not converge: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    return EstimationResult(
        converged=result.success,
        iterations=result.nit,
        fmin=result.fun,
        theta=result.x,
        hessian_inv=getattr(result, "hess_inv", None),
        sample_cov=sample_cov,
        n_obs=n_obs,
        spec=spec,
    )
