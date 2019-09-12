from typing import Any, Optional

import matplotlib.pyplot as plt
from typing_extensions import Protocol


class SupportsPlot(Protocol):
    """A class of objects that knows how to plot itself to an axes."""

    def _plot_(self, ax: plt.Axes, **kwargs) -> Any:
        raise NotImplementedError


def plot(obj: SupportsPlot, ax: Optional[plt.Axes] = None, **kwargs) -> Any:
    """Plots an object to a given Axes or a new Axes and show it.

    Args:
        obj: an object with a _plot_() method that knows how to plot itself
            to an axes.
        ax: if given, plot onto it. Otherwise, create a new Axes.
        kwargs: additional arguments passed to obj._plot_().
    Returns:
        A 2-tuple:
          - The Axes that's plotted on.
          - The return value of obj._plot_().
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    result = obj._plot_(ax, **kwargs)
    ax.get_figure().show(warn=False)
    return ax, result
