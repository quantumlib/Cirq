# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool to visualize the results of a study."""

from typing import Union, Optional, Sequence, SupportsFloat
import collections
import numpy as np
import matplotlib.pyplot as plt
import cirq.study.result as result


def get_state_histogram(result: 'result.Result') -> np.ndarray:
    """Computes a state histogram from a single result with repetitions.

    Args:
        result: The trial result containing measurement results from which the
                state histogram should be computed.

    Returns:
        The state histogram (a numpy array of length 2 ** num_qubits)
        corresponding to the trial result. The state histogram is computed
        using `result.histogram()`.
    """
    num_qubits = sum([value.shape[1] for value in result.measurements.values()])
    hist = result.histogram(key=','.join(result.data.keys()))
    data = np.zeros(2 ** num_qubits)
    for k, v in hist.items():
        data[k] = v
    return data


def plot_state_histogram(
    data: Union['result.Result', collections.Counter, Sequence[SupportsFloat]],
    ax: Optional['plt.Axis'] = None,
    *,
    tick_label: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = 'qubit state',
    ylabel: Optional[str] = 'result count',
    title: Optional[str] = 'Result State Histogram',
) -> 'plt.Axis':
    """Plot the state histogram from either a single result with repetitions or
       a histogram of measurement results computed using `result.histogram()` or
       a histogram of measurement results specified as an np.ndarray where i'th
       entry corresponds to the histogram of state |i>

    Args:
        data:   The histogram values to plot. Possible options are:
                `result.Result`: Histogram is computed using
                `get_state_histogram` and all 2 ** num_qubits values are
                 plotted, including 0s.
                `collections.Counter`: Only (key, value) pairs present in
                 collection are plotted.
                `Sequence[SupportsFloat]`: Values in input are plotted with
                 default labels as |0> ... |n-1>.
        ax:      The Axes to plot on. If not given, a new figure is created,
                 plotted on, and shown.
        tick_label: Tick labels for the histogram plot. If not given, the keys
                    of `collections.Counter` are used by default. For sequence,
                    label for i'th entry is |i>.
        xlabel:  Label for the x-axis.
        ylabel:  Label for the y-axis.
        title:   Title of the plot.

    Returns:
        The axis that was plotted on.
    """
    show_fig = not ax
    if not ax:
        fig, ax = plt.subplots(1, 1)
    if isinstance(data, result.Result):
        data = get_state_histogram(data)
    labels = np.arange(len(data))
    if isinstance(data, collections.Counter):
        labels, data = zip(*sorted(data.items()))
    if not tick_label:
        tick_label = labels
    ax.bar(np.arange(len(data)), data, tick_label=tick_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_fig:
        fig.show()
    return ax
