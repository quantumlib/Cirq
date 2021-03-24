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

from typing import TYPE_CHECKING
import numpy as np
from cirq._compat import deprecated

if TYPE_CHECKING:
    from cirq.study import result


@deprecated(
    deadline="v0.12",
    fix="use cirq.vis.plot_state_histogram or cirq.vis.get_state_histogram instead",
    name="cirq.study.visualize.plot_state_histogram",
)
def plot_state_histogram(result: 'result.Result') -> np.ndarray:
    """Plot the state histogram from a single result with repetitions.

    States is a bitstring representation of all the qubit states in a single
    result.

    Args:
        result: The trial results to plot.

    Returns:
        The histogram. A list of values plotted on the y-axis.
    """
    # Needed to avoid circular imports.
    import cirq.vis as vis

    values = vis.get_state_histogram(result)
    vis.plot_state_histogram(values)
    return values
