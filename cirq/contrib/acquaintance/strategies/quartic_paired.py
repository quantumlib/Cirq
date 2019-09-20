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

from typing import cast, Iterable, List, Sequence, Tuple, TYPE_CHECKING

from cirq import circuits, ops
from cirq.contrib.acquaintance.devices import (
    UnconstrainedAcquaintanceDevice)
from cirq.contrib.acquaintance.gates import acquaint, SwapNetworkGate
from cirq.contrib.acquaintance.mutation_utils import (
    expose_acquaintance_gates)

if TYPE_CHECKING:
    import cirq


def qubit_pairs_to_qubit_order(qubit_pairs: Sequence[Sequence['cirq.Qid']]
                              ) -> List['cirq.Qid']:
    """Takes a sequence of qubit pairs and returns a sequence in which every
    pair is at distance two.

    Specifically, given pairs (1a, 1b), (2a, 2b), etc. returns
    (1a, 2a, 1b, 2b, 3a, 4a, 3b, 4b, ...).
    """

    if set(len(qubit_pair) for qubit_pair in qubit_pairs) != set((2,)):
        raise ValueError(
            'set(len(qubit_pair) for qubit_pair in qubit_pairs) != '
            'set((2,))')
    n_pairs = len(qubit_pairs)
    qubits = []  # type: List['cirq.Qid']
    for i in range(0, 2 * (n_pairs // 2), 2):
        qubits += [qubit_pairs[i][0], qubit_pairs[i + 1][0],
                   qubit_pairs[i][1], qubit_pairs[i + 1][1]]
    if n_pairs % 2:
        qubits += list(qubit_pairs[-1])
    return qubits


def quartic_paired_acquaintance_strategy(
        qubit_pairs: Iterable[Tuple['cirq.Qid', ops.Qid]]
) -> Tuple['cirq.Circuit', Sequence['cirq.Qid']]:
    """Acquaintance strategy for pairs of pairs.

    Implements UpCCGSD ansatz from arXiv:1810.02327.
    """

    qubit_pairs = tuple(
        cast(Tuple['cirq.Qid', ops.Qid], tuple(qubit_pair))
        for qubit_pair in qubit_pairs)
    qubits = qubit_pairs_to_qubit_order(qubit_pairs)
    n_qubits = len(qubits)
    swap_network = SwapNetworkGate((1,) * n_qubits, 2)(*qubits)
    strategy = circuits.Circuit(swap_network,
                                device=UnconstrainedAcquaintanceDevice)
    expose_acquaintance_gates(strategy)
    for i in reversed(range(0, n_qubits, 2)):
        moment = ops.Moment([acquaint(*qubits[j: j + 4])
                             for j in range(i % 4, n_qubits - 3, 4)])
        strategy.insert(2 * i, moment)
    return strategy, qubits
