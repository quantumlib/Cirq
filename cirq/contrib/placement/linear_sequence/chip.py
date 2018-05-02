# Copyright 2018 Google LLC
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

from typing import Dict, List, Tuple

from cirq.google import XmonDevice, XmonQubit


EDGE = Tuple[XmonQubit, XmonQubit]


def above(qubit: XmonQubit) -> XmonQubit:
    """Gives qubit with one unit less on the second coordinate.

    Args:
      qubit: Reference qubit.

    Returns:
      New translated qubit.
    """
    return XmonQubit(qubit.row, qubit.col - 1)


def left_of(qubit: XmonQubit) -> XmonQubit:
    """Gives qubit with one unit less on the first coordinate.

    Args:
      qubit: Reference qubit.

    Returns:
      New translated qubit.
    """
    return XmonQubit(qubit.row - 1, qubit.col)


def below(qubit: XmonQubit) -> XmonQubit:
    """Gives qubit with one unit more on the second coordinate.

    Args:
      qubit: Reference qubit.

    Returns:
      New translated qubit.
    """
    return XmonQubit(qubit.row, qubit.col + 1)


def right_of(qubit: XmonQubit) -> XmonQubit:
    """Gives node with one unit more on the first coordinate.

    Args:
      node: Reference node.

    Returns:
      New translated node.
    """
    return XmonQubit(qubit.row + 1, qubit.col)


def yx_cmp(n: XmonQubit, m: XmonQubit) -> int:
    """Comparator that compares first by second and then first coordinate.

    Args:
      n: Left operand.
      m: Right operand.

    Returns:
      0 if qubits are equal, negative number if qubit n is less than m, and
      positive otherwise.
    """
    if n.col != m.col:
        return (n.col > m.col) - (n.col < m.col)
    else:
        return (n.row > m.row) - (n.row < m.row)


def chip_as_adjacency_list(device: XmonDevice) -> Dict[
    XmonQubit, List[XmonQubit]]:
    """Gives adjacency list representation of a chip.

    The adjacency list is constructed in order of above, left_of, below and
    right_of consecutively.

    Args:
      c: Chip to be converted.

    Returns:
      Map from nodes to list of qubits which represent all the neighbours of
      given qubit.
    """
    c_set = set(device.qubits)
    c_adj = {} # type: Dict[XmonQubit, List[XmonQubit]] 
    for n in device.qubits:
        c_adj[n] = []
        for m in [above(n), left_of(n), below(n), right_of(n)]:
            if m in c_set:
                c_adj[n].append(m)
    return c_adj
