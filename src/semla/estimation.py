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
    sample_mean: np.ndarray = None
    raw_data: np.ndarray = None  # centered raw data (for MLR Gamma)


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


def _model_implied_mean(
    A: np.ndarray, m: np.ndarray, F: np.ndarray
) -> np.ndarray | None:
    """Compute model-implied mean vector: mu = F @ inv(I-A) @ m."""
    n = A.shape[0]
    I = np.eye(n)
    try:
        IminA_inv = np.linalg.inv(I - A)
    except np.linalg.LinAlgError:
        return None
    return F @ IminA_inv @ m


def ml_objective(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    sample_mean: np.ndarray = None,
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
    sample_mean : array, optional
        Sample mean vector (p,). Required when spec.meanstructure=True.

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

        # Mean structure contribution
        if spec.meanstructure and sample_mean is not None:
            m = spec.unpack_m(theta)
            mu = _model_implied_mean(A, m, spec.F)
            if mu is None:
                return 1e10
            diff = sample_mean - mu
            fml += float(diff @ sigma_inv @ diff)

        return fml

    except np.linalg.LinAlgError:
        return 1e10


def ml_gradient(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    sample_mean: np.ndarray = None,
) -> np.ndarray:
    """Compute the analytic gradient of F_ML.

    Uses the RAM derivative formulas:
        dF/dA_{rc} = (B^T G C + B G C)[r,c] + mean terms
        dF/dS_{rc} = (B^T G B)[r,c] (×2 for off-diagonal)

    where B = (I-A)^{-1}, G = F^T W F, W = Σ^{-1} - Σ^{-1}(S+dd^T)Σ^{-1},
    C = B S B^T.
    """
    A, S_mat = spec.unpack(theta)
    n = A.shape[0]
    I = np.eye(n)

    try:
        B = np.linalg.inv(I - A)
    except np.linalg.LinAlgError:
        return _ml_gradient_numerical(theta, spec, sample_cov, n_obs, sample_mean)

    C = B @ S_mat @ B.T
    sigma = spec.F @ C @ spec.F.T

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return _ml_gradient_numerical(theta, spec, sample_cov, n_obs, sample_mean)

    # W matrix (covariance part)
    W_arg = sample_cov.copy()
    d = None
    if spec.meanstructure and sample_mean is not None:
        m = spec.unpack_m(theta)
        mu = _model_implied_mean(A, m, spec.F)
        if mu is not None:
            d = sample_mean - mu
            W_arg = W_arg + np.outer(d, d)

    W = sigma_inv - sigma_inv @ W_arg @ sigma_inv
    G = spec.F.T @ W @ spec.F

    # Gradient matrix for A parameters: D_A = 2 * B^T @ G @ C
    GC = G @ C
    D_A = 2.0 * B.T @ GC

    # Gradient matrix for S parameters: D_S = B^T @ G @ B
    D_S = B.T @ G @ B

    # Mean gradient contributions
    D_A_mean = None
    D_m_mean = None
    if d is not None:
        q = B.T @ spec.F.T @ sigma_inv @ d
        P = B @ m
        D_A_mean = -2.0 * np.outer(q, P)
        D_m_mean = -2.0 * q

    # Extract raw gradient in parameter order
    # 1. A_free (row-major)
    grad_A = D_A[spec.A_free]
    if D_A_mean is not None:
        grad_A = grad_A + D_A_mean[spec.A_free]

    # 2. S_free_lower (lower triangle, row-major)
    S_lower = spec._S_free_lower
    grad_S_raw = D_S.copy()
    # Off-diagonal elements need factor of 2
    for r in range(n):
        for c in range(r):
            if S_lower[r, c]:
                grad_S_raw[r, c] = 2.0 * D_S[r, c]
    grad_S = grad_S_raw[S_lower]

    # 3. m_free (if meanstructure)
    parts = [grad_A, grad_S]
    if spec.meanstructure and spec.m_free is not None and D_m_mean is not None:
        grad_m = D_m_mean[spec.m_free]
        parts.append(grad_m)

    raw_grad = np.concatenate(parts)

    # Apply constraint map (sum gradients for equal-constrained params)
    if spec._constraint_map is not None:
        n_eff = int(spec._constraint_map.max()) + 1
        eff_grad = np.zeros(n_eff)
        for i in range(len(raw_grad)):
            eff_grad[spec._constraint_map[i]] += raw_grad[i]
        return eff_grad

    return raw_grad


def _ml_gradient_numerical(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    sample_mean: np.ndarray = None,
) -> np.ndarray:
    """Fallback: compute gradient via finite differences."""
    eps = 1e-7
    grad = np.zeros_like(theta)
    f0 = ml_objective(theta, spec, sample_cov, n_obs, sample_mean)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (ml_objective(theta_plus, spec, sample_cov, n_obs, sample_mean) - f0) / eps
    return grad


def _compute_se(
    theta: np.ndarray,
    spec: ModelSpecification,
    sample_cov: np.ndarray,
    n_obs: int,
    return_vcov: bool = False,
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
            # Covariance structure information
            cov_info = 0.5 * (n_obs - 1) * np.trace(SinvdS[i] @ SinvdS[j])
            info[i, j] = cov_info
            info[j, i] = cov_info

    # Mean structure information (additive, block-diagonal under normality)
    if spec.meanstructure:
        # Compute dmu/dtheta_i numerically
        A0, _ = spec.unpack(theta)
        m0 = spec.unpack_m(theta)
        mu0 = _model_implied_mean(A0, m0, spec.F)
        if mu0 is not None:
            dmu = np.zeros((k, p))
            for i in range(k):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[i] += eps
                theta_minus[i] -= eps
                A_p, _ = spec.unpack(theta_plus)
                A_m, _ = spec.unpack(theta_minus)
                m_p = spec.unpack_m(theta_plus)
                m_m = spec.unpack_m(theta_minus)
                mu_p = _model_implied_mean(A_p, m_p, spec.F)
                mu_m = _model_implied_mean(A_m, m_m, spec.F)
                if mu_p is not None and mu_m is not None:
                    dmu[i] = (mu_p - mu_m) / (2 * eps)

            # I_ij(mean) = (N-1) * dmu_i' @ Sigma^{-1} @ dmu_j
            for i in range(k):
                for j in range(i, k):
                    mean_info = (n_obs - 1) * float(dmu[i] @ sigma_inv @ dmu[j])
                    info[i, j] += mean_info
                    if i != j:
                        info[j, i] += mean_info

    # Var(theta) = inv(I), SE = sqrt(diag(inv(I)))
    try:
        info_inv = np.linalg.inv(info)
        var_theta = np.diag(info_inv)
        se = np.where(var_theta > 0, np.sqrt(var_theta), np.nan)
        if return_vcov:
            return se, info_inv
        return se
    except np.linalg.LinAlgError:
        if return_vcov:
            return np.full(k, np.nan), np.full((k, k), np.nan)
        return np.full(k, np.nan)


def estimate(
    spec: ModelSpecification,
    data: pd.DataFrame,
    scipy_constraints: list[dict] | None = None,
) -> EstimationResult:
    """Estimate model parameters via Maximum Likelihood.

    Parameters
    ----------
    spec : ModelSpecification
        Model specification from ``build_specification()``.
    data : pd.DataFrame
        Data with columns matching observed variables.
    scipy_constraints : list[dict], optional
        Nonlinear constraints in scipy format. When present, SLSQP is used
        instead of BFGS.

    Returns
    -------
    EstimationResult
    """
    # Compute sample statistics
    obs_data = data[spec.observed_vars].values
    n_obs = obs_data.shape[0]
    sample_cov = np.cov(obs_data, rowvar=False, ddof=1)

    # Make sure it's 2D even with 1 variable
    if sample_cov.ndim == 0:
        sample_cov = sample_cov.reshape(1, 1)

    sample_mean = None
    if spec.meanstructure:
        sample_mean = np.mean(obs_data, axis=0)

    # Starting values
    theta0 = spec.pack_start()

    if scipy_constraints:
        # Constrained optimization with SLSQP
        result = optimize.minimize(
            ml_objective,
            theta0,
            args=(spec, sample_cov, n_obs, sample_mean),
            method="SLSQP",
            jac=ml_gradient,
            constraints=scipy_constraints,
            options={"maxiter": 10000, "ftol": 1e-10},
        )

        # Polish with tighter tolerance
        if result.success:
            result2 = optimize.minimize(
                ml_objective,
                result.x,
                args=(spec, sample_cov, n_obs, sample_mean),
                method="SLSQP",
                jac=ml_gradient,
                constraints=scipy_constraints,
                options={"maxiter": 10000, "ftol": 1e-12},
            )
            if result2.fun <= result.fun + 1e-10:
                result2.success = True
                result = result2
    else:
        # Unconstrained optimization with BFGS
        result = optimize.minimize(
            ml_objective,
            theta0,
            args=(spec, sample_cov, n_obs, sample_mean),
            method="BFGS",
            jac=ml_gradient,
            options={"maxiter": 10000, "gtol": 1e-6},
        )

        # Polish: re-run from the BFGS solution with tighter tolerance
        if result.success:
            result2 = optimize.minimize(
                ml_objective,
                result.x,
                args=(spec, sample_cov, n_obs, sample_mean),
                method="BFGS",
                jac=ml_gradient,
                options={"maxiter": 10000, "gtol": 1e-9},
            )
            if result2.fun <= result.fun + 1e-10:
                result2.success = True  # accept if objective didn't increase
                result = result2

    if not result.success:
        # Compute diagnostic info
        grad = ml_gradient(result.x, spec, sample_cov, n_obs, sample_mean)
        grad_norm = float(np.linalg.norm(grad))
        msg = (
            f"Model did not converge after {result.nit} iterations.\n"
            f"  Diagnostic: gradient norm = {grad_norm:.2e}, "
            f"f(x) = {result.fun:.6f}\n"
            f"  Possible causes:\n"
            f"  - Model may be misspecified (check syntax and path diagram)\n"
            f"  - Model may be empirically underidentified\n"
            f"  - Try different starting values or add constraints"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Center data for potential MLR Gamma computation
    centered_data = obs_data - obs_data.mean(axis=0)

    return EstimationResult(
        converged=result.success,
        iterations=result.nit,
        fmin=result.fun,
        theta=result.x,
        hessian_inv=getattr(result, "hess_inv", None),
        sample_cov=sample_cov,
        n_obs=n_obs,
        spec=spec,
        sample_mean=sample_mean,
        raw_data=centered_data,
    )
