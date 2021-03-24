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

from typing import Union, Optional
import math
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
    states = 2 ** num_qubits
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
    values: Union['result.Result', np.ndarray], ax: Optional['plt.Axis'] = None
) -> 'plt.Axis':
    """Plot the state histogram from either a single result with repetitions or
       a histogram of measurement results computed using `get_state_histogram`.

    Args:
        values: The histogram values to plot. If `result.Result` is passed, the
                values are computed by calling `get_state_histogram`.
        ax:     The Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.

    Returns:
        The axis that was plotted on.
    """
    show_fig = not ax
    if not ax:
        fig, ax = plt.subplots(1, 1)
    print(values, isinstance(values, result.Result))
    if isinstance(values, result.Result):
        values = get_state_histogram(values)
    states = len(values)
    num_qubits = math.ceil(math.log(states, 2))
    plot_labels = [bin(x)[2:].zfill(num_qubits) for x in range(states)]
    ax.bar(np.arange(states), values, tick_label=plot_labels)
    ax.set_xlabel('qubit state')
    ax.set_ylabel('result count')
    if show_fig:
        fig.show()
    return ax
