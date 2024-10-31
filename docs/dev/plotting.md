# Plotting guidelines

Here we recommend the input arguments, return value, and behavior of the
`plot` method of a class.

## Requirements

1. **Convenience to interactive users.** This is the highest priority.
    Compared to being called in a batch script as a library (for composing
    more complicated plots or other purposes), the `plot` method is mainly
    used in interactive sessions like ipython, jupyter, colab, PyCharm,
    and python interpreter.
1. **Plot is customizable.** The plot should be customizable by the user after
    `plot` returns. This is important because user may need to change the look
    for presentation, paper, or just the style they prefer. One plot style does
    not fit all.
1. **No unnecessary messages in interactive sessions.** It should not produce
    any warning/error messages in normal usages in an interactive sessions.
    See [#1890](https://github.com/quantumlib/Cirq/issues/1890#issue-473510953)
    for an example of such message.
1. **No popups during tests.** It should not produce any pop-up windows during
    tests.

## Recommendation

The `plot` method must produce a plot when there is no arguments in an
interactive session. The recommended way to achieve that is illustrated in the
example below.

```python
from typing import Any, List, Optional
import matplotlib.pyplot as plt

class Foo:
    ...
    def plot(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
        show_plot = not ax
        if ax is None:
            fig, ax = plt.subplots(1, 1)  # or your favorite figure setup
        # Call methods of the ax instance like ax.plot to plot on it.
        ...
        if show_plot:
            fig.show()
        return ax
```

This `plot` method works in 2 modes: *memory mode* and *interactive mode*,
signalled by the presence of the `ax` argument. When present, the method is
instructed to plot on the provided `ax` instance in memory. No plot is shown
on the screen. When absent, the code is in *interactive* mode, and it creates
a figure and shows it.

The returned `ax` instance can be used to further customize the plot if the
user wants to. Note that if we were to call `plt.show` instead of `fig.show`,
the customizations on the returned `ax` does not show up on subsequent call to
`plt.show`.

To satisfy requirement number 4, unit test codes should create an `ax` object
and pass it into the `plot` method like the following example.

```python
def test_foo_plot():
    # make a Foo instance foo
    figure, ax = plt.subplots(1, 1)
    foo.plot(ax)
    # assert on the content of ax here if necessary.
```

This does not produce a pop-up window because `fig.show` is not called.


## Classes that produce multi-axes plot

Some classes contain complicated data and plotting on a single `ax` is
not sufficient. The `plot` method of such a class should take an optional
`axes` argument that is a list of `plt.Axes` instances.

```python
class Foo:
    ...
    def plot(self, axes: Optional[List[plt.Axes]]=None,
             **plot_kwargs: Any) -> List[plt.Axes]:
        show_plot = not axes
        if axes is None:
            fig, axes = plt.subplots(1, 2)  # or your favorite figure setup
        elif len(axes) != 2:  # your required number of axes
            raise ValueError('your error message')
        # Call methods of the axes[i] objects to plot on it.
        ...
        if show_plot:
            fig.show()
        return axes
```

The reason we don't recommend passing a `plt.Figure` argument is that, the
`plot` method has no information on which `plt.Axes` objects to plot on if
there are more `plt.Axes` in the figure than what the method needs. The caller
is responsible for passing in correct number of `Axes` instances.

The `plot` method can be tested similarly.

## PyCharm issue

As of this writing in October 2019, running a script calling a `plot` method
in PyCharm does not pop up a window with the figure. A call to `plt.show()`
is needed to show it. We believe this is a PyCharm-specific issue because
the same code works in Python interpreter.

## References
* Issue #1890 "Plotting code should not call `show`"
* PR #2097
* PR #2286
