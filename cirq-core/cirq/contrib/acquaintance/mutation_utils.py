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

from __future__ import annotations

import collections
from typing import cast, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from cirq import circuits, ops, transformers
from cirq.contrib.acquaintance.devices import get_acquaintance_size
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate, SwapNetworkGate
from cirq.contrib.acquaintance.permutation import PermutationGate

if TYPE_CHECKING:
    import cirq

STRATEGY_GATE = Union[AcquaintanceOpportunityGate, PermutationGate]


def rectify_acquaintance_strategy(circuit: cirq.Circuit, acquaint_first: bool = True) -> None:
    """Splits moments so that they contain either only acquaintance or permutation gates.

    Orders resulting moments so that the first one is of the same type as the previous one.

    Args:
        circuit: The acquaintance strategy to rectify.
        acquaint_first: Whether to make acquaintance moment first in when
        splitting the first mixed moment.

    Raises:
        TypeError: If the circuit is not an acquaintance strategy.
    """
    rectified_moments = []
    for moment in circuit:
        gate_type_to_ops: Dict[bool, List[ops.GateOperation]] = collections.defaultdict(list)
        for op in moment.operations:
            gate_op = cast(ops.GateOperation, op)
            is_acquaintance = isinstance(gate_op.gate, AcquaintanceOpportunityGate)
            gate_type_to_ops[is_acquaintance].append(gate_op)
        if len(gate_type_to_ops) == 1:
            rectified_moments.append(moment)
            continue
        for acquaint_first in sorted(gate_type_to_ops.keys(), reverse=acquaint_first):
            rectified_moments.append(circuits.Moment(gate_type_to_ops[acquaint_first]))
    circuit._moments = rectified_moments


def replace_acquaintance_with_swap_network(
    circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
    acquaintance_size: Optional[int] = 0,
    swap_gate: cirq.Gate = ops.SWAP,
) -> bool:
    """Replace every rectified moment with acquaintance gates with a generalized swap network.

    The generalized swap network has a partition given by the acquaintance gates in that moment
    (and singletons for the free qubits). Accounts for reversing effect of swap networks.

    Args:
        circuit: The acquaintance strategy.
        qubit_order: The qubits, in order, on which the replacing swap network
            gate acts on.
        acquaintance_size: The acquaintance size of the new swap network gate.
        swap_gate: The gate used to swap logical indices.

    Returns: Whether or not the overall effect of the inserted swap network
        gates is to reverse the order of the qubits, i.e. the parity of the
        number of swap network gates inserted.

    Raises:
        TypeError: circuit is not an acquaintance strategy.
    """
    rectify_acquaintance_strategy(circuit)
    reflected = False
    reverse_map = {q: r for q, r in zip(qubit_order, reversed(qubit_order))}
    for moment_index, moment in enumerate(circuit):
        if reflected:
            moment = moment.transform_qubits(reverse_map.__getitem__)
        if all(isinstance(op.gate, AcquaintanceOpportunityGate) for op in moment.operations):
            swap_network_gate = SwapNetworkGate.from_operations(
                qubit_order, moment.operations, acquaintance_size, swap_gate
            )
            swap_network_op = swap_network_gate(*qubit_order)
            moment = circuits.Moment([swap_network_op])
            reflected = not reflected
        circuit._moments[moment_index] = moment
    return reflected


class ExposeAcquaintanceGates:
    """Decomposes permutation gates that provide acquaintance opportunities."""

    def __init__(self):
        self.no_decomp = lambda op: (
            not get_acquaintance_size(op) or isinstance(op.gate, AcquaintanceOpportunityGate)
        )

    def optimize_circuit(self, circuit: cirq.Circuit) -> None:
        circuit._moments = [*transformers.expand_composite(circuit, no_decomp=self.no_decomp)]

    def __call__(self, circuit: cirq.Circuit) -> None:
        self.optimize_circuit(circuit)


expose_acquaintance_gates = ExposeAcquaintanceGates()
