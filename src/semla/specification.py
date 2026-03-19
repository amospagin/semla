"""Translate parsed syntax into RAM matrix specification for estimation.

Uses the Reticular Action Model (RAM) notation:
    A — asymmetric paths (directed: loadings, regressions)
    S — symmetric paths (variances, covariances)
    F — filter matrix (selects observed variables from full variable vector)

Model-implied covariance:
    Sigma = F @ (I - A)^{-1} @ S @ ((I - A)^{-1})^T @ F^T
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .syntax import FormulaToken


@dataclass
class ParamInfo:
    """Metadata for a single parameter in the model."""

    lhs: str
    op: str
    rhs: str
    free: bool
    value: float  # starting value or fixed value
    label: str | None = None


@dataclass
class ModelSpecification:
    """RAM matrix specification built from parsed syntax tokens."""

    observed_vars: list[str]
    latent_vars: list[str]
    all_vars: list[str]  # observed + latent

    params: list[ParamInfo] = field(default_factory=list)

    # RAM matrices (filled by build())
    A_free: np.ndarray = field(default=None)  # bool mask: which cells are free
    A_values: np.ndarray = field(default=None)  # starting/fixed values
    S_free: np.ndarray = field(default=None)
    S_values: np.ndarray = field(default=None)
    F: np.ndarray = field(default=None)  # filter matrix

    # Mean structure (optional)
    m_free: np.ndarray = field(default=None)   # bool mask (n_vars,)
    m_values: np.ndarray = field(default=None)  # starting/fixed values (n_vars,)
    meanstructure: bool = False

    # Equality constraints: maps raw free param index -> effective theta index
    # When None, no constraints (1:1 mapping). When set, same-labeled
    # params map to the same effective index.
    _constraint_map: np.ndarray = field(default=None)

    @property
    def n_obs(self) -> int:
        return len(self.observed_vars)

    @property
    def n_latent(self) -> int:
        return len(self.latent_vars)

    @property
    def n_vars(self) -> int:
        return len(self.all_vars)

    @property
    def _S_free_lower(self) -> np.ndarray:
        """Boolean mask for free S parameters, lower triangle only."""
        return np.tril(self.S_free)

    @property
    def _n_free_raw(self) -> int:
        """Number of free parameters before applying equality constraints."""
        count = int(np.sum(self.A_free) + np.sum(self._S_free_lower))
        if self.meanstructure and self.m_free is not None:
            count += int(np.sum(self.m_free))
        return count

    @property
    def n_free(self) -> int:
        """Number of unique free parameters (after equality constraints)."""
        if self._constraint_map is not None:
            return int(self._constraint_map.max()) + 1
        return self._n_free_raw

    def _idx(self, var: str) -> int:
        return self.all_vars.index(var)

    def pack_start(self) -> np.ndarray:
        """Pack free parameter starting values into a 1-D vector."""
        a_vals = self.A_values[self.A_free]
        s_vals = self.S_values[self._S_free_lower]
        parts = [a_vals, s_vals]
        if self.meanstructure and self.m_free is not None:
            parts.append(self.m_values[self.m_free])
        raw = np.concatenate(parts)

        if self._constraint_map is not None:
            # Collapse to effective theta (take first value per group)
            n_eff = int(self._constraint_map.max()) + 1
            theta = np.zeros(n_eff)
            for i in range(len(raw)):
                eff_idx = self._constraint_map[i]
                theta[eff_idx] = raw[i]  # last write wins (all same-label have same start)
            return theta
        return raw

    def _expand_theta(self, theta_effective: np.ndarray) -> np.ndarray:
        """Expand effective (constrained) theta to raw theta."""
        if self._constraint_map is not None:
            return theta_effective[self._constraint_map]
        return theta_effective

    def param_theta_index(self, p: "ParamInfo") -> int | None:
        """Return the index in the effective theta vector for a given free parameter.

        Accounts for equality constraints.
        Returns None if the parameter is fixed.
        """
        raw_idx = self._param_raw_theta_index(p)
        if raw_idx is None:
            return None
        if self._constraint_map is not None:
            return int(self._constraint_map[raw_idx])
        return raw_idx

    def _param_raw_theta_index(self, p: "ParamInfo") -> int | None:
        """Return the raw (pre-constraint) theta index."""
        if not p.free:
            return None

        if p.op == "=~":
            row = self._idx(p.rhs)
            col = self._idx(p.lhs)
            # Count how many True entries in A_free come before (row, col) in row-major
            flat_idx = row * self.n_vars + col
            a_flat = self.A_free.ravel()
            count = 0
            for k in range(flat_idx):
                if a_flat[k]:
                    count += 1
            return count

        elif p.op == "~":
            row = self._idx(p.lhs)
            col = self._idx(p.rhs)
            flat_idx = row * self.n_vars + col
            a_flat = self.A_free.ravel()
            count = 0
            for k in range(flat_idx):
                if a_flat[k]:
                    count += 1
            return count

        elif p.op == "~~":
            n_a = int(np.sum(self.A_free))
            i = self._idx(p.lhs)
            j = self._idx(p.rhs)
            # Use lower triangle: row = max, col = min
            row, col = max(i, j), min(i, j)
            s_lower = self._S_free_lower
            # Count how many True entries come before (row, col) in row-major
            count = 0
            for r in range(self.n_vars):
                for c in range(r + 1):  # lower triangle: c <= r
                    if r == row and c == col:
                        return n_a + count
                    if s_lower[r, c]:
                        count += 1
            return None

        elif p.op == "~1":
            if not self.meanstructure or self.m_free is None:
                return None
            n_a = int(np.sum(self.A_free))
            n_s = int(np.sum(self._S_free_lower))
            i = self._idx(p.lhs)
            count = int(np.sum(self.m_free[:i]))
            return n_a + n_s + count

        return None

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack a 1-D parameter vector into A and S matrices."""
        raw = self._expand_theta(theta)
        n_a = int(np.sum(self.A_free))
        n_s = int(np.sum(self._S_free_lower))
        A = self.A_values.copy()
        S = self.S_values.copy()
        A[self.A_free] = raw[:n_a]
        S[self._S_free_lower] = raw[n_a:n_a + n_s]
        # Mirror to upper triangle for symmetry
        S = np.tril(S) + np.tril(S, -1).T
        return A, S

    def unpack_m(self, theta: np.ndarray) -> np.ndarray:
        """Unpack the m (intercept) vector from theta.

        Only meaningful when meanstructure=True.
        """
        raw = self._expand_theta(theta)
        n_a = int(np.sum(self.A_free))
        n_s = int(np.sum(self._S_free_lower))
        m = self.m_values.copy()
        if self.m_free is not None:
            m[self.m_free] = raw[n_a + n_s:]
        return m


def build_specification(
    tokens: list[FormulaToken],
    observed_columns: list[str],
    auto_var: bool = True,
    auto_cov_latent: bool = True,
    fixed_x: bool = False,
    meanstructure: bool = False,
    int_ov_free: bool = True,
    int_lv_free: bool = False,
) -> ModelSpecification:
    """Build a RAM specification from parsed tokens and data column names.

    Parameters
    ----------
    tokens : list[FormulaToken]
        Output of ``parse_syntax()``.
    observed_columns : list[str]
        Column names from the data (observed variables).
    auto_var : bool
        Automatically add residual variances for all variables.
    auto_cov_latent : bool
        Automatically add covariances between latent variables (CFA default).
    fixed_x : bool
        If True, treat exogenous observed variables as fixed (not estimated).
    meanstructure : bool
        If True, estimate intercepts for observed variables.
    int_ov_free : bool
        If True (default), observed-variable intercepts are freely estimated.
        Set to False for growth curve models where observed intercepts are
        fixed to 0.
    int_lv_free : bool
        If True, latent-variable intercepts (means) are freely estimated.
        Set to True for growth curve models to estimate intercept and slope
        means.
    sample_means : np.ndarray, optional
        Sample means for observed variables (used as starting values).

    Returns
    -------
    ModelSpecification
    """
    # Auto-detect mean structure from syntax
    if any(tok.op == "~1" for tok in tokens):
        meanstructure = True
    # Identify latent variables (appear on LHS of =~)
    latent_vars: list[str] = []
    for tok in tokens:
        if tok.op == "=~" and tok.lhs not in latent_vars:
            latent_vars.append(tok.lhs)

    # Collect all observed vars referenced in the model
    referenced_obs: list[str] = []
    for tok in tokens:
        for term in tok.rhs:
            if term.var in observed_columns and term.var not in referenced_obs:
                referenced_obs.append(term.var)
        # LHS of ~ or ~~ might also be observed
        if tok.op in ("~", "~~") and tok.lhs in observed_columns:
            if tok.lhs not in referenced_obs:
                referenced_obs.append(tok.lhs)

    observed_vars = referenced_obs
    all_vars = observed_vars + latent_vars
    n = len(all_vars)

    spec = ModelSpecification(
        observed_vars=observed_vars,
        latent_vars=latent_vars,
        all_vars=all_vars,
    )

    A_free = np.zeros((n, n), dtype=bool)
    A_values = np.zeros((n, n), dtype=float)
    S_free = np.zeros((n, n), dtype=bool)
    S_values = np.zeros((n, n), dtype=float)

    # Filter matrix: selects observed from all_vars
    F = np.zeros((len(observed_vars), n), dtype=float)
    for i, ov in enumerate(observed_vars):
        F[i, all_vars.index(ov)] = 1.0

    params: list[ParamInfo] = []

    # --- Process tokens ---
    first_indicator: dict[str, bool] = {}  # track first indicator per latent

    for tok in tokens:
        if tok.op == "=~":
            # Factor loadings: A[indicator, latent] = loading
            lv = tok.lhs
            for term in tok.rhs:
                row = spec._idx(term.var)
                col = spec._idx(lv)

                if lv not in first_indicator:
                    # Fix first loading to 1.0 for identification
                    first_indicator[lv] = True
                    if term.fixed:
                        val = term.start_value if term.start_value is not None else 1.0
                        A_values[row, col] = val
                    elif term.modifier is None:
                        # Default: fix first to 1.0
                        A_values[row, col] = 1.0
                    else:
                        # Has a label but no fixed value — still fix to 1
                        A_values[row, col] = 1.0
                    params.append(ParamInfo(lv, "=~", term.var, free=False, value=A_values[row, col]))
                else:
                    if term.fixed:
                        val = term.start_value if term.start_value is not None else 0.0
                        A_values[row, col] = val
                        params.append(ParamInfo(lv, "=~", term.var, free=False, value=val))
                    else:
                        A_free[row, col] = True
                        A_values[row, col] = 0.5  # starting value
                        label = term.modifier if isinstance(term.modifier, str) else None
                        params.append(ParamInfo(lv, "=~", term.var, free=True, value=0.5, label=label))

        elif tok.op == "~":
            # Regression: A[dv, iv] = coefficient
            dv = tok.lhs
            for term in tok.rhs:
                row = spec._idx(dv)
                col = spec._idx(term.var)
                if term.fixed:
                    val = term.start_value if term.start_value is not None else 0.0
                    A_values[row, col] = val
                    params.append(ParamInfo(dv, "~", term.var, free=False, value=val))
                else:
                    A_free[row, col] = True
                    A_values[row, col] = 0.0
                    label = term.modifier if isinstance(term.modifier, str) else None
                    params.append(ParamInfo(dv, "~", term.var, free=True, value=0.0, label=label))

        elif tok.op == "~~":
            # (Co)variance: S[var1, var2] = value
            for term in tok.rhs:
                i = spec._idx(tok.lhs)
                j = spec._idx(term.var)
                if term.fixed:
                    val = term.start_value if term.start_value is not None else 0.0
                    S_values[i, j] = val
                    S_values[j, i] = val
                    params.append(ParamInfo(tok.lhs, "~~", term.var, free=False, value=val))
                else:
                    S_free[i, j] = True
                    S_free[j, i] = True
                    start = 1.0 if i == j else 0.0
                    S_values[i, j] = start
                    S_values[j, i] = start
                    label = term.modifier if isinstance(term.modifier, str) else None
                    params.append(ParamInfo(tok.lhs, "~~", term.var, free=True, value=start, label=label))

    # --- Auto-add residual variances ---
    if auto_var:
        for var in all_vars:
            i = spec._idx(var)
            if not S_free[i, i] and S_values[i, i] == 0.0:
                S_free[i, i] = True
                S_values[i, i] = 0.5  # starting value for variances
                params.append(ParamInfo(var, "~~", var, free=True, value=0.5))

    # --- Auto-add covariances between latent variables (CFA) ---
    if auto_cov_latent and len(latent_vars) > 1:
        for i_idx in range(len(latent_vars)):
            for j_idx in range(i_idx + 1, len(latent_vars)):
                lv_i = latent_vars[i_idx]
                lv_j = latent_vars[j_idx]
                ii = spec._idx(lv_i)
                jj = spec._idx(lv_j)
                if not S_free[ii, jj] and S_values[ii, jj] == 0.0:
                    S_free[ii, jj] = True
                    S_free[jj, ii] = True
                    S_values[ii, jj] = 0.05
                    S_values[jj, ii] = 0.05
                    params.append(ParamInfo(lv_i, "~~", lv_j, free=True, value=0.05))

    spec.A_free = A_free
    spec.A_values = A_values
    spec.S_free = S_free
    spec.S_values = S_values
    spec.F = F

    # --- Mean structure ---
    if meanstructure:
        m_free = np.zeros(n, dtype=bool)
        m_values = np.zeros(n, dtype=float)

        # Process explicit ~1 tokens
        explicit_intercepts = set()
        for tok in tokens:
            if tok.op == "~1":
                i = spec._idx(tok.lhs)
                m_free[i] = True
                m_values[i] = 0.0  # starting value (overridden by Model with sample means)
                explicit_intercepts.add(tok.lhs)
                params.append(ParamInfo(tok.lhs, "~1", "1", free=True, value=0.0))

        # Auto-add intercepts for observed variables not explicitly specified
        if int_ov_free:
            for var in observed_vars:
                if var not in explicit_intercepts:
                    i = spec._idx(var)
                    m_free[i] = True
                    m_values[i] = 0.0  # starting value (overridden by Model with sample means)
                    params.append(ParamInfo(var, "~1", "1", free=True, value=0.0))

        # Latent variable intercepts: fixed to 0 by default, freed for growth models
        if int_lv_free:
            for lv in latent_vars:
                if lv not in explicit_intercepts:
                    i = spec._idx(lv)
                    m_free[i] = True
                    m_values[i] = 0.0
                    params.append(ParamInfo(lv, "~1", "1", free=True, value=0.0))

        spec.m_free = m_free
        spec.m_values = m_values
        spec.meanstructure = True

    spec.params = params

    # --- Equality constraints from labels ---
    # Find free params with the same label and create a constraint map
    labeled_params = [(i, p) for i, p in enumerate(params) if p.free and p.label]
    label_groups: dict[str, list[int]] = {}
    for i, p in labeled_params:
        label_groups.setdefault(p.label, []).append(i)

    # Only create constraint map if there are actual shared labels
    has_constraints = any(len(indices) > 1 for indices in label_groups.values())
    if has_constraints:
        # Build raw theta index for each free param
        free_params = [(i, p) for i, p in enumerate(params) if p.free]
        raw_indices = {}
        for raw_idx, (param_idx, p) in enumerate(free_params):
            raw_indices[param_idx] = raw_idx

        n_raw = len(free_params)
        constraint_map = np.arange(n_raw, dtype=int)  # default: identity

        # For each label group, map all to the same effective index
        effective_idx = 0
        assigned = np.full(n_raw, -1, dtype=int)

        # First pass: assign effective indices
        for raw_idx in range(n_raw):
            if assigned[raw_idx] >= 0:
                continue
            param_idx = free_params[raw_idx][0]
            p = free_params[raw_idx][1]

            assigned[raw_idx] = effective_idx

            # If this param has a label, find all others with same label
            if p.label and p.label in label_groups:
                for other_param_idx in label_groups[p.label]:
                    if other_param_idx in raw_indices:
                        other_raw = raw_indices[other_param_idx]
                        if assigned[other_raw] < 0:
                            assigned[other_raw] = effective_idx

            effective_idx += 1

        # Handle any unassigned (shouldn't happen but be safe)
        for raw_idx in range(n_raw):
            if assigned[raw_idx] < 0:
                assigned[raw_idx] = effective_idx
                effective_idx += 1

        spec._constraint_map = assigned

    return spec
