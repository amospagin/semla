"""semla — Structural Equation Modeling with lavaan-style syntax for Python."""

from ._version import __version__
from .model import Model, MultiGroupModel, cfa, sem, growth
from .comparisons import chi_square_diff_test
from .diagnostics import mardia_test
from .irt import irt, IRTModel
from . import datasets
from . import priors


def set_host_devices(n: int) -> None:
    """Set the number of CPU devices for parallel MCMC chains.

    Call this at the top of your script, **before** any Bayesian model
    fitting.  Once JAX initializes, the device count is locked.

    Parameters
    ----------
    n : int
        Number of CPU devices (typically equal to ``chains``).
    """
    from .bayes import set_host_devices as _set
    _set(n)

__all__ = [
    "Model", "MultiGroupModel", "cfa", "sem", "growth", "irt", "IRTModel",
    "chi_square_diff_test", "mardia_test", "datasets", "priors",
    "set_host_devices", "__version__",
]
