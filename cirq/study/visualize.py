# Copyright 2018 The Cirq Developers
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

import numpy as np

from cirq.study import trial_result


def plot_state_histogram(result: trial_result.TrialResult) -> np.ndarray:
    """Plot the state histogram from a single result with repetitions.

    States is a bitstring representation of all the qubit states in a single
    result.
    Currently this function assumes each measurement gate applies to only
    a single qubit.

    Args:
        result: The trial results to plot.

    Returns:
        The histogram. A list of values plotted on the y-axis.
    """

    # pyplot import is deferred because it requires a system dependency
    # (python3-tk) that `pip install cirq` can't handle for the user. This
    # allows cirq to be usable without python3-tk.
    import matplotlib.pyplot as plt

    num_qubits = len(result.measurements.keys())
    states = 2**num_qubits
    values = np.zeros(states)

    # measurements is a dict of {measurement gate key:
    #                            array(repetitions, boolean result)}
    # Convert this to an array of repetitions, each with an array of booleans.
    # e.g. {q1: array([[True, True]]), q2: array([[False, False]])}
    #      --> array([[True, False], [True, False]])
    measurement_by_result = np.array([
        v.transpose()[0] for k, v in result.measurements.items()]).transpose()

    for meas in measurement_by_result:
        # Convert each array of booleans to a string representation.
        # e.g. [True, False] -> [1, 0] -> '10' -> 2
        state_ind = int(''.join([str(x) for x in [int(x) for x in meas]]), 2)
        values[state_ind] += 1

    plot_labels = [bin(x)[2:].zfill(num_qubits) for x in range(states)]
    plt.bar(np.arange(states), values, tick_label=plot_labels)
    plt.xlabel('qubit state')
    plt.ylabel('result count')
    plt.show()

    return values
