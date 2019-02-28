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

import functools
from typing import cast, Callable, Iterable, Optional

from cirq import circuits, ops
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.mutation_utils import (
        replace_acquaintance_with_swap_network)
from cirq.contrib.acquaintance.permutation import SwapPermutationGate
from cirq.contrib.acquaintance.devices import UnconstrainedAcquaintanceDevice


def quartic_bipartite_acquaintance_strategy(
        first_qubits: Iterable[ops.QubitId],
        second_qubits: Iterable[ops.QubitId],
        swap_gate: ops.Gate = ops.SWAP,
        acquainter: Optional[Callable[[circuits.Circuit], None]] = None
        ) -> circuits.Circuit:
    """
    Returns an acquaintance strategy capable of executing a gate corresponding
    to any 4-tuple of qubits that contains a pair of qubits from one set and a
    pair of qubits from another.

    Args:
        first_qubits: The first set of qubits.
        second_qubits: The second set of qubits.
        swap_gate: The swap gate to use.

    Returns:
        A circuit capable of implementing any operation on a pair of pairs of
        qubits.
    """

    qubit_sets = tuple(first_qubits), tuple(second_qubits)

    if min(len(qubit_set) for qubit_set in qubit_sets) < 2:
        raise ValueError('min(len(qubit_set) '
                         'for qubit_set in qubit_sets) < 2')

    if set(qubit_sets[0]) & set(qubit_sets[1]):
        raise ValueError('set(first_qubits) & set(second_qubits)')

    qubit_set_sizes = tuple(len(qubit_set) for qubit_set in qubit_sets)
    qubits = qubit_sets[0] + qubit_sets[1]

    moments = []
    for first_layer in range(qubit_set_sizes[0]):
        first_acquaintances = tuple(
                acquaint(*qubit_sets[0][i: i + 2])
                for i in range(first_layer % 2, qubit_set_sizes[0] - 1, 2))
        first_swaps = tuple(
            SwapPermutationGate(swap_gate)(*qubit_sets[0][i: i + 2])
            for i in range(first_layer % 2, qubit_set_sizes[0] - 1, 2))
        for second_layer in range(qubit_set_sizes[1]):
            second_acquaintances = tuple(
                acquaint(*qubit_sets[1][i: i + 2])
                for i in range(second_layer % 2, qubit_set_sizes[1] - 1, 2))
            moment = ops.Moment(first_acquaintances + second_acquaintances)
            moments.append(moment)
            second_swaps = tuple(
                SwapPermutationGate(swap_gate)(*qubit_sets[1][i: i + 2])
                for i in range(second_layer % 2, qubit_set_sizes[1] - 1, 2))
            moment = (ops.Moment(first_swaps + second_swaps) if
                      (second_layer == qubit_set_sizes[1] - 1) else
                      ops.Moment(second_swaps))
            moments.append(moment)
    strategy = circuits.Circuit(moments, device=UnconstrainedAcquaintanceDevice)
    if acquainter is None:
        acquainter = cast(Callable[[circuits.Circuit], None],
                functools.partial(replace_acquaintance_with_swap_network,
                qubit_order=qubits, acquaintance_size=None))
    acquainter(strategy)
    return strategy
