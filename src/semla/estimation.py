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
    """Compute standard errors from numerically estimated Hessian."""
    eps = 1e-5
    k = len(theta)
    H = np.zeros((k, k))
    f0 = ml_objective(theta, spec, sample_cov, n_obs)

    for i in range(k):
        for j in range(i, k):
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()

            theta_pp[i] += eps
            theta_pp[j] += eps
            theta_pm[i] += eps
            theta_pm[j] -= eps
            theta_mp[i] -= eps
            theta_mp[j] += eps
            theta_mm[i] -= eps
            theta_mm[j] -= eps

            H[i, j] = (
                ml_objective(theta_pp, spec, sample_cov, n_obs)
                - ml_objective(theta_pm, spec, sample_cov, n_obs)
                - ml_objective(theta_mp, spec, sample_cov, n_obs)
                + ml_objective(theta_mm, spec, sample_cov, n_obs)
            ) / (4 * eps * eps)
            H[j, i] = H[i, j]

    # Standard errors = sqrt(diag(2 * inv(H) / (n-1)))
    # Factor of 2 because F_ML = 2 * log-likelihood ratio / (n-1)
    # Actually: Var(theta) = 2 * inv(H) / (N - 1) for normal-theory ML
    try:
        H_inv = np.linalg.inv(H)
        var_theta = 2.0 * np.diag(H_inv) / (n_obs - 1)
        # Protect against negative variances
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

    # Optimize
    result = optimize.minimize(
        ml_objective,
        theta0,
        args=(spec, sample_cov, n_obs),
        method="BFGS",
        jac=ml_gradient,
        options={"maxiter": 5000, "gtol": 1e-6},
    )

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
