# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, TYPE_CHECKING

from cirq.devices import GridQubit

if TYPE_CHECKING:
    import cirq_google


EDGE = Tuple[GridQubit, GridQubit]


def above(qubit: GridQubit) -> GridQubit:
    """Gives qubit with one unit less on the second coordinate.

    Args:
        qubit: Reference qubit.

    Returns:
        New translated qubit.
    """
    return GridQubit(qubit.row, qubit.col - 1)


def left_of(qubit: GridQubit) -> GridQubit:
    """Gives qubit with one unit less on the first coordinate.

    Args:
        qubit: Reference qubit.

    Returns:
        New translated qubit.
    """
    return GridQubit(qubit.row - 1, qubit.col)


def below(qubit: GridQubit) -> GridQubit:
    """Gives qubit with one unit more on the second coordinate.

    Args:
        qubit: Reference qubit.

    Returns:
        New translated qubit.
    """
    return GridQubit(qubit.row, qubit.col + 1)


def right_of(qubit: GridQubit) -> GridQubit:
    """Gives node with one unit more on the first coordinate.

    Args:
        qubit: Reference node.

    Returns:
        New translated node.
    """
    return GridQubit(qubit.row + 1, qubit.col)


def chip_as_adjacency_list(
    device: 'cirq_google.XmonDevice',
) -> Dict[GridQubit, List[GridQubit]]:
    """Gives adjacency list representation of a chip.

    The adjacency list is constructed in order of above, left_of, below and
    right_of consecutively.

    Args:
        device: Chip to be converted.

    Returns:
        Map from nodes to list of qubits which represent all the neighbours of
        given qubit.
    """
    c_set = set(device.qubits)
    c_adj = {}  # type: Dict[GridQubit, List[GridQubit]]
    for n in device.qubits:
        c_adj[n] = []
        for m in [above(n), left_of(n), below(n), right_of(n)]:
            if m in c_set:
                c_adj[n].append(m)
    return c_adj
