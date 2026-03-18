"""semla — Structural Equation Modeling with lavaan-style syntax for Python."""

from ._version import __version__
from .model import Model, MultiGroupModel, cfa, sem
from .comparisons import chi_square_diff_test
from . import datasets

__all__ = [
    "Model", "MultiGroupModel", "cfa", "sem",
    "chi_square_diff_test", "datasets", "__version__",
]
