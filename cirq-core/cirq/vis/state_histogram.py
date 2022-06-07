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
        The state histogram (a numpy array) corresponding to the trial result.
    """
    num_qubits = sum([value.shape[1] for value in result.measurements.values()])
    states = 2**num_qubits
    values = np.zeros(states)
    # measurements is a dict of {measurement gate key:
    #                            array(repetitions, boolean result)}
    # Convert this to an array of repetitions, each with an array of booleans.
    # e.g. {q1: array([[True, True]]), q2: array([[False, False]])}
    #      --> array([[True, False], [True, False]])
    measurement_by_result = np.hstack(list(result.measurements.values()))

    for meas in measurement_by_result:
        # Convert each array of booleans to a string representation.
        # e.g. [True, False] -> [1, 0] -> '10' -> 2
        state_ind = int(''.join([str(x) for x in [int(x) for x in meas]]), 2)
        values[state_ind] += 1
    return values


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
       a histogram computed using `result.histogram()` or a flattened histogram
       of measurement results computed using `get_state_histogram`.

    Args:
        data:   The histogram values to plot. Possible options are:
                `result.Result`: Histogram is computed using
                    `get_state_histogram` and all 2 ** num_qubits values are
                    plotted, including 0s.
                `collections.Counter`: Only (key, value) pairs present in
                    collection are plotted.
                `Sequence[SupportsFloat]`: Values in the input sequence are
                    plotted. i'th entry corresponds to height of the i'th
                    bar in histogram.
        ax:      The Axes to plot on. If not given, a new figure is created,
                 plotted on, and shown.
        tick_label: Tick labels for the histogram plot in case input is not
                    `collections.Counter`. By default, label for i'th entry
                     is |i>.
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
        values = get_state_histogram(data)
    elif isinstance(data, collections.Counter):
        tick_label, values = zip(*sorted(data.items()))
    else:
        values = np.array(data)
    if not tick_label:
        tick_label = np.arange(len(values))
    ax.bar(np.arange(len(values)), values, tick_label=tick_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_fig:
        fig.show()
    return ax
