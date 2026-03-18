"""Built-in datasets for semla."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent


def HolzingerSwineford1939() -> pd.DataFrame:
    """Load the Holzinger and Swineford (1939) dataset.

    This is the classic dataset used in lavaan tutorials and SEM textbooks.
    It contains mental ability test scores for 301 seventh- and eighth-grade
    children from two schools (Pasteur and Grant-White).

    Variables
    ---------
    id : int
        Student identifier.
    sex : int
        1 = male, 2 = female.
    ageyr : int
        Age in years.
    agemo : int
        Age in months (remainder after years).
    school : str
        School name ("Pasteur" or "Grant-White").
    grade : int
        Grade level (7 or 8).
    x1 : float
        Visual perception.
    x2 : float
        Cubes.
    x3 : float
        Lozenges.
    x4 : float
        Paragraph comprehension.
    x5 : float
        Sentence completion.
    x6 : float
        Word meaning.
    x7 : float
        Speeded addition.
    x8 : float
        Speeded counting of dots.
    x9 : float
        Speeded discrimination straight and curved capitals.

    Returns
    -------
    pd.DataFrame
        DataFrame with 301 rows and 15 columns.

    References
    ----------
    Holzinger, K. J., & Swineford, F. A. (1939). A study in factor analysis:
    The stability of a bi-factor solution. Supplementary Educational
    Monographs, No. 48. University of Chicago Press.

    Examples
    --------
    >>> from semla.datasets import HolzingerSwineford1939
    >>> df = HolzingerSwineford1939()
    >>> df.shape
    (301, 15)
    """
    return pd.read_csv(_DATA_DIR / "HolzingerSwineford1939.csv")
