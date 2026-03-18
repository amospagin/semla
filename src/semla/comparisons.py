"""Model comparison utilities."""

from __future__ import annotations

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
