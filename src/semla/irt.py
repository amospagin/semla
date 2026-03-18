"""Item Response Theory (IRT) models via CFA parameterization.

IRT models are mathematically equivalent to single-factor CFA:
    2PL: discrimination = loading, difficulty = -threshold/loading
    1PL (Rasch): 2PL with equal discriminations
    GRM: single-factor CFA with ordinal items
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Model, cfa
from .syntax import parse_syntax


@dataclass
class IRTParams:
    """IRT parameterization of a fitted model."""

    item: str
    discrimination: float  # a parameter
    difficulty: float | list[float]  # b parameter (single for 2PL, list for GRM)
    se_discrimination: float = np.nan
    se_difficulty: float | list[float] = np.nan


class IRTModel:
    """Item Response Theory model.

    Wraps a CFA model and provides IRT-specific parameterization,
    item/test information functions, and ability estimation.

    Parameters
    ----------
    items : list[str]
        Item column names.
    data : pd.DataFrame
        Data with item responses (binary or ordinal).
    model_type : str
        ``"1PL"``, ``"2PL"``, or ``"GRM"`` (Graded Response Model).
    """

    def __init__(self, items: list[str], data: pd.DataFrame,
                 model_type: str = "2PL", **kwargs):
        self.items = items
        self.data = data
        self.model_type = model_type.upper()

        if self.model_type not in ("1PL", "2PL", "GRM"):
            raise ValueError(
                f"model_type must be '1PL', '2PL', or 'GRM', got '{model_type}'"
            )

        # Build CFA syntax
        if self.model_type == "1PL":
            # Equal discriminations via label constraint
            item_str = " + ".join(f"a*{item}" for item in items)
        else:
            item_str = " + ".join(items)

        syntax = f"ability =~ {item_str}"
        self._cfa_fit = cfa(syntax, data=data, estimator="DWLS", **kwargs)

        # Extract thresholds from polychoric computation
        self._compute_irt_params()

    @property
    def converged(self) -> bool:
        return self._cfa_fit.converged

    def _compute_irt_params(self):
        """Convert CFA parameters to IRT parameterization."""
        est = self._cfa_fit.estimates()
        std = self._cfa_fit.standardized_estimates("std.all")

        self._irt_params = []
        for item in self.items:
            # Get unstandardized loading
            row = est[(est["op"] == "=~") & (est["rhs"] == item)]
            if len(row) == 0:
                continue

            loading = row["est"].values[0]
            loading_se = row["se"].values[0]

            # Get standardized loading for discrimination conversion
            std_row = std[(std["op"] == "=~") & (std["rhs"] == item)]
            std_loading = std_row["est.std"].values[0] if len(std_row) > 0 else loading

            # Discrimination: a = lambda_std / sqrt(1 - lambda_std^2)
            # This converts from factor analysis metric to IRT metric
            if abs(std_loading) < 0.999:
                discrimination = std_loading / np.sqrt(1 - std_loading ** 2)
            else:
                discrimination = np.sign(std_loading) * 10.0  # cap at extreme

            # SE of discrimination via delta method
            # d(a)/d(lambda) = 1 / (1 - lambda^2)^(3/2)
            if not np.isnan(loading_se) and abs(std_loading) < 0.999:
                da_dl = 1.0 / (1 - std_loading ** 2) ** 1.5
                se_disc = abs(da_dl) * loading_se
            else:
                se_disc = np.nan

            # Difficulty: b = -tau / lambda (unstandardized)
            # For binary items, there's one threshold
            # For ordinal items, multiple thresholds
            # We estimate difficulty from the sample proportions
            item_data = self.data[item].dropna()
            categories = sorted(item_data.unique())

            if len(categories) == 2:
                # Binary: single difficulty
                p = (item_data == categories[1]).mean()
                p = np.clip(p, 0.001, 0.999)
                # Threshold in probit metric
                tau = -np.log(p / (1 - p)) / 1.7  # approximate probit
                difficulty = tau / discrimination if abs(discrimination) > 0.01 else np.nan
                self._irt_params.append(IRTParams(
                    item=item,
                    discrimination=discrimination,
                    difficulty=difficulty,
                    se_discrimination=se_disc,
                ))
            else:
                # Ordinal: multiple difficulties (one per category boundary)
                cum_props = np.cumsum([(item_data == c).mean() for c in categories])
                difficulties = []
                for cp in cum_props[:-1]:
                    cp = np.clip(cp, 0.001, 0.999)
                    tau = -np.log(cp / (1 - cp)) / 1.7
                    b = tau / discrimination if abs(discrimination) > 0.01 else np.nan
                    difficulties.append(b)
                self._irt_params.append(IRTParams(
                    item=item,
                    discrimination=discrimination,
                    difficulty=difficulties,
                    se_discrimination=se_disc,
                ))

    def irt_params(self) -> pd.DataFrame:
        """Return IRT parameters (discrimination and difficulty).

        Returns
        -------
        pd.DataFrame
            Columns: item, discrimination, difficulty, se_discrimination.
            For GRM, difficulty is the first threshold difficulty.
        """
        rows = []
        for p in self._irt_params:
            if isinstance(p.difficulty, list):
                # For display, show all thresholds
                for k, b in enumerate(p.difficulty):
                    rows.append({
                        "item": p.item,
                        "threshold": k + 1,
                        "discrimination": p.discrimination,
                        "difficulty": b,
                        "se_discrimination": p.se_discrimination,
                    })
            else:
                rows.append({
                    "item": p.item,
                    "threshold": 1,
                    "discrimination": p.discrimination,
                    "difficulty": p.difficulty,
                    "se_discrimination": p.se_discrimination,
                })
        return pd.DataFrame(rows)

    def icc(self, theta: np.ndarray = None) -> pd.DataFrame:
        """Compute Item Characteristic Curves.

        Parameters
        ----------
        theta : np.ndarray, optional
            Ability values. Defaults to np.linspace(-3, 3, 61).

        Returns
        -------
        pd.DataFrame
            Columns: theta, item1_prob, item2_prob, ...
        """
        if theta is None:
            theta = np.linspace(-3, 3, 61)

        result = {"theta": theta}
        for p in self._irt_params:
            a = p.discrimination
            if isinstance(p.difficulty, list):
                # GRM: P(X >= k) for first threshold
                b = p.difficulty[0] if p.difficulty else 0.0
            else:
                b = p.difficulty

            if np.isnan(b):
                result[p.item] = np.full_like(theta, np.nan)
            else:
                # 2PL ICC: P(X=1|theta) = 1 / (1 + exp(-1.7*a*(theta - b)))
                result[p.item] = 1.0 / (1.0 + np.exp(-1.7 * a * (theta - b)))

        return pd.DataFrame(result)

    def item_information(self, theta: np.ndarray = None) -> pd.DataFrame:
        """Compute Item Information Functions.

        I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))
        (for 2PL; scaled by 1.7^2 for normal ogive metric)

        Parameters
        ----------
        theta : np.ndarray, optional
            Ability values.

        Returns
        -------
        pd.DataFrame
            Columns: theta, item1_info, item2_info, ...
        """
        icc_df = self.icc(theta)
        theta_vals = icc_df["theta"].values
        result = {"theta": theta_vals}

        for p in self._irt_params:
            prob = icc_df[p.item].values
            a = p.discrimination
            info = (1.7 * a) ** 2 * prob * (1 - prob)
            result[p.item] = info

        return pd.DataFrame(result)

    def test_information(self, theta: np.ndarray = None) -> pd.DataFrame:
        """Compute Test Information Function.

        I(theta) = sum_i I_i(theta)

        Returns
        -------
        pd.DataFrame
            Columns: theta, information, se (standard error of theta).
        """
        item_info = self.item_information(theta)
        theta_vals = item_info["theta"].values
        total_info = sum(item_info[p.item].values for p in self._irt_params)
        se_theta = 1.0 / np.sqrt(np.maximum(total_info, 1e-10))

        return pd.DataFrame({
            "theta": theta_vals,
            "information": total_info,
            "se": se_theta,
        })

    def abilities(self, method: str = "regression") -> pd.DataFrame:
        """Estimate person ability (theta) scores.

        Equivalent to factor scores from the underlying CFA.

        Parameters
        ----------
        method : str
            ``"regression"`` or ``"bartlett"``.

        Returns
        -------
        pd.DataFrame
            Single column 'ability' with theta estimates.
        """
        scores = self._cfa_fit.predict(method=method)
        return scores.rename(columns={"ability": "theta"})

    def summary(self) -> str:
        """Print IRT model summary."""
        lines = []
        lines.append(f"semla IRT — {self.model_type} Model")
        lines.append("=" * 50)
        lines.append(f"  Items: {len(self.items)}")
        lines.append(f"  Observations: {len(self.data)}")
        lines.append(f"  Converged: {self.converged}")
        lines.append("")

        # Fit indices from underlying CFA
        idx = self._cfa_fit.fit_indices()
        lines.append("Model Fit:")
        lines.append(f"  Chi-square: {idx['chi_square']:.3f} (df={idx['df']})")
        lines.append(f"  CFI: {idx['cfi']:.3f}")
        lines.append(f"  RMSEA: {idx['rmsea']:.3f}")
        lines.append("")

        # IRT parameters
        lines.append("Item Parameters:")
        lines.append(f"  {'Item':<15} {'Discrim':>8} {'Difficulty':>10}")
        lines.append("  " + "-" * 35)
        for p in self._irt_params:
            if isinstance(p.difficulty, list):
                b_str = ", ".join(f"{b:.3f}" for b in p.difficulty)
                lines.append(f"  {p.item:<15} {p.discrimination:>8.3f} {b_str:>10}")
            else:
                b_str = f"{p.difficulty:.3f}" if not np.isnan(p.difficulty) else "NA"
                lines.append(f"  {p.item:<15} {p.discrimination:>8.3f} {b_str:>10}")
        lines.append("")

        # Reliability
        rel = self._cfa_fit.reliability()
        if "ability" in rel:
            lines.append(f"  Omega: {rel['ability']['omega']:.3f}")
            lines.append(f"  Alpha: {rel['ability']['alpha']:.3f}")

        lines.append("=" * 50)
        output = "\n".join(lines)
        print(output)
        return output


def irt(items: list[str], data: pd.DataFrame, model_type: str = "2PL",
        **kwargs) -> IRTModel:
    """Fit an IRT model.

    Parameters
    ----------
    items : list[str]
        Item column names in the data.
    data : pd.DataFrame
        Data with item responses.
    model_type : str
        ``"1PL"``, ``"2PL"``, or ``"GRM"``.

    Returns
    -------
    IRTModel

    Examples
    --------
    >>> fit = irt(["item1", "item2", "item3"], data=df, model_type="2PL")
    >>> fit.irt_params()
    >>> fit.abilities()
    """
    return IRTModel(items, data, model_type=model_type, **kwargs)
