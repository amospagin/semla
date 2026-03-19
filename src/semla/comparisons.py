"""Model comparison utilities."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def chi_square_diff_test(model_restricted, model_free) -> dict:
    """Chi-square difference test between nested models.

    Parameters
    ----------
    model_restricted : Model or MultiGroupModel
        More constrained model (higher df).
    model_free : Model or MultiGroupModel
        Less constrained model (lower df).

    Returns
    -------
    dict
        chi_sq_diff, df_diff, p_value.
    """
    r = model_restricted.results
    f = model_free.results

    delta_chi = r.chi_square - f.chi_square
    delta_df = r.df - f.df

    if delta_df <= 0:
        raise ValueError(
            f"Restricted model must have more df than free model. "
            f"Got restricted df={r.df}, free df={f.df}."
        )

    p_value = 1.0 - stats.chi2.cdf(delta_chi, delta_df)

    return {
        "chi_sq_diff": delta_chi,
        "df_diff": delta_df,
        "p_value": p_value,
    }


def compare_models(**models) -> pd.DataFrame:
    """Compare multiple models in an anova-style table.

    Parameters
    ----------
    **models : Model or MultiGroupModel
        Named models to compare. Pass as keyword arguments, e.g.
        ``compare_models(configural=fit1, metric=fit2, scalar=fit3)``.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by df, with columns for chi-square, df,
        AIC, BIC, CFI, RMSEA, and delta chi-square tests between
        successive models.

    Examples
    --------
    >>> compare_models(configural=fit1, metric=fit2, scalar=fit3)
    """
    if len(models) < 2:
        raise ValueError("At least 2 models are required for comparison.")

    rows = []
    for name, model in models.items():
        fi = model.fit_indices()
        rows.append({
            "model": name,
            "chisq": fi.get("chi_square", float("nan")),
            "df": fi.get("df", 0),
            "pvalue": fi.get("p_value", float("nan")),
            "aic": fi.get("aic", float("nan")),
            "bic": fi.get("bic", float("nan")),
            "cfi": fi.get("cfi", float("nan")),
            "rmsea": fi.get("rmsea", float("nan")),
            "srmr": fi.get("srmr", float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("df").reset_index(drop=True)

    # Compute delta chi-square between successive rows
    delta_chisq = [float("nan")]
    delta_df = [float("nan")]
    delta_p = [float("nan")]
    for i in range(1, len(df)):
        d_chi = df.iloc[i]["chisq"] - df.iloc[i - 1]["chisq"]
        d_df = df.iloc[i]["df"] - df.iloc[i - 1]["df"]
        delta_chisq.append(d_chi)
        delta_df.append(d_df)
        if d_df > 0:
            delta_p.append(1.0 - stats.chi2.cdf(d_chi, d_df))
        else:
            delta_p.append(float("nan"))

    df["delta_chisq"] = delta_chisq
    df["delta_df"] = delta_df
    df["delta_pvalue"] = delta_p

    df = df.set_index("model")
    return df
