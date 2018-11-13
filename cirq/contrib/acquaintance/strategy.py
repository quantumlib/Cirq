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

import collections

from typing import Sequence, TYPE_CHECKING, Union

from cirq import circuits, ops, optimizers

from cirq.contrib.acquaintance.gates import (
     SwapNetworkGate, AcquaintanceOpportunityGate, ACQUAINT)
from cirq.contrib.acquaintance.permutation import (
        PermutationGate)
from cirq.contrib.acquaintance.devices import (
    UnconstrainedAcquaintanceDevice,
    is_acquaintance_strategy, get_acquaintance_size)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Type, Dict, List
    from cirq.ops import Gate

STRATEGY_GATE = Union[AcquaintanceOpportunityGate, PermutationGate]


def rectify_acquaintance_strategy(
        circuit: circuits.Circuit,
        acquaint_first: bool=True
        ) -> None:
    """Splits moments so that they contain either only acquaintance gates
    or only permutation gates. Orders resulting moments so that the first one
    is of the same type as the previous one.

    Args:
        circuit: The acquaintance strategy to rectify.
        acquaint_first: Whether to make acquaintance moment first in when
        splitting the first mixed moment.
    """

    if not is_acquaintance_strategy(circuit):
        raise TypeError('not is_acquaintance_strategy(circuit)')

    rectified_moments = []
    for moment in circuit:
        gate_type_to_ops = collections.defaultdict(list
                ) # type: Dict[bool, List[ops.GateOperation]]
        for op in moment.operations:
            gate_type_to_ops[isinstance(op.gate, AcquaintanceOpportunityGate)
                    ].append(op)
        if len(gate_type_to_ops) == 1:
            rectified_moments.append(moment)
            continue
        for acquaint_first in sorted(gate_type_to_ops.keys(),
                                     reverse=acquaint_first):
            rectified_moments.append(
                    circuits.Moment(gate_type_to_ops[acquaint_first]))
    circuit._moments = rectified_moments


def replace_acquaintance_with_swap_network(
        circuit: circuits.Circuit,
        qubit_order: Sequence[ops.QubitId],
        acquaintance_size: int=0
        ) -> bool:
    """
    Replace every moment containing acquaintance gates (after
    rectification) with a generalized swap network, with the partition
    given by the acquaintance gates in that moment (and singletons for the
    free qubits). Accounts for reversing effect of swap networks.

    Args:
        circuit: The acquaintance strategy.
        qubit_order: The qubits, in order, on which the replacing swap network
            gate acts on.
        acquaintance_size: The acquaintance size of the new swap network gate.

    Returns: Whether or not the overall effect of the inserted swap network
        gates is to reverse the order of the qubits, i.e. the parity of the
        number of swap network gates inserted.

    Raises:
        TypeError: circuit is not an acquaintance strategy.
    """

    if not is_acquaintance_strategy(circuit):
        raise TypeError('not is_acquaintance_strategy(circuit)')

    rectify_acquaintance_strategy(circuit)
    reflected = False
    reverse_map = {q: r for q, r in zip(qubit_order, reversed(qubit_order))}
    for moment_index, moment in enumerate(circuit):
        if reflected:
            moment = moment.transform_qubits(reverse_map.__getitem__)
        if all(isinstance(op.gate, AcquaintanceOpportunityGate)
                for op in moment.operations):
            swap_network_gate = SwapNetworkGate.from_operations(
                    qubit_order, moment.operations, acquaintance_size)
            swap_network_op = swap_network_gate(*qubit_order)
            moment = circuits.Moment([swap_network_op])
            reflected = not reflected
        circuit._moments[moment_index] = moment
    return reflected


class ExposeAcquaintanceGates(optimizers.ExpandComposite):
    """Decomposes any permutation gates that provide acquaintance opportunities
    in order to make them explicit."""
    def __init__(self):
        circuits.PointOptimizer.__init__(self)
        self.no_decomp = lambda op: (
                not get_acquaintance_size(op) or
                (isinstance(op, ops.GateOperation) and
                 isinstance(op.gate, AcquaintanceOpportunityGate)))


expose_acquaintance_gates = ExposeAcquaintanceGates()


def complete_acquaintance_strategy(qubit_order: Sequence[ops.QubitId],
                                   acquaintance_size: int=0,
                                   ) -> circuits.Circuit:
    """
    Returns an acquaintance strategy capable of executing a gate corresponding
    to any set of at most acquaintance_size qubits.

    Args:
        qubit_order: The qubits on which the strategy should be defined.
        acquaintance_size: The maximum number of qubits to be acted on by
        an operation.

    Returns:
        An circuit capable of implementing any set of k-local
        operation.
    """
    if acquaintance_size < 0:
        raise ValueError('acquaintance_size must be non-negative.')
    elif acquaintance_size == 0:
        return circuits.Circuit(device=UnconstrainedAcquaintanceDevice)

    strategy = circuits.Circuit.from_ops(
            (ACQUAINT(q) for q in qubit_order),
            device=UnconstrainedAcquaintanceDevice)
    for size_to_acquaint in range(2, acquaintance_size + 1):
        expose_acquaintance_gates(strategy)
        replace_acquaintance_with_swap_network(
                strategy, qubit_order, size_to_acquaint)
    return strategy
