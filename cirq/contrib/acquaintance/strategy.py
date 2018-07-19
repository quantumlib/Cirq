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

from collections import defaultdict
from functools import partial
from operator import ne
from typing import Sequence, TYPE_CHECKING, Union

from cirq.circuits import Circuit, Moment, ExpandComposite
from cirq.extension import Extensions
from cirq.ops import QubitId, GateOperation
from cirq.contrib.acquaintance.gates import (
     SwapNetworkGate, AcquaintanceOpportunityGate, ACQUAINT,
     op_acquaintance_size)
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
        PermutationGate, LinearPermutationGate)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Type, Dict, List
    from cirq.ops import Gate

STRATEGY_GATE = Union[AcquaintanceOpportunityGate, PermutationGate]

class AcquaintanceStrategy(Circuit):
    gate_types = (AcquaintanceOpportunityGate, PermutationGate)

    def __init__(self, circuit: Circuit=None) -> None:
        if circuit is None:
            circuit = Circuit()
        for moment in circuit:
            for op in moment.operations:
                if not isinstance(op.gate, self.gate_types):
                    raise ValueError('An acquaintance strategy can only'
                        'contain gates of types {}'.format(self.gate_types))
        super().__init__(circuit._moments)

    def rectify(self, acquaint_first: bool=True):
        last_gate_type = self.gate_types[int(not acquaint_first)]
        rectified_moments = []
        for moment in self:
            gate_type_to_ops = defaultdict(list
                    ) # type: Dict[Type[STRATEGY_GATE], List[GateOperation]]
            for op in moment.operations:
                gate_type_to_ops[type(op.gate)].append(op)
            if len(gate_type_to_ops) == 1:
                rectified_moments.append(moment)
                continue
            for gate_type in sorted(gate_type_to_ops,
                                    key=partial(ne, last_gate_type)):
                rectified_moments.append(Moment(gate_type_to_ops[gate_type]))
                last_gate_type = gate_type
        self._moments = rectified_moments

    def nest(self, qubit_order: Sequence[QubitId], acquaintance_size: int=0
            ) -> bool:
        self.rectify()
        reflected = False
        reverse_map = {q: r for q, r in zip(qubit_order, reversed(qubit_order))}
        for moment_index, moment in enumerate(self):
            if reflected:
                moment = moment.with_qubits_mapped(reverse_map)
            if all(isinstance(op.gate, AcquaintanceOpportunityGate)
                    for op in moment.operations):
                swap_network_gate = SwapNetworkGate.from_operations(
                        qubit_order, moment.operations, acquaintance_size)
                swap_network_op = swap_network_gate(*qubit_order)
                self._moments[moment_index] = Moment([swap_network_op])
                reflected = not reflected
        return reflected

    def acquaintance_size(self) -> int:
        return max(op_acquaintance_size(op) for op in self.all_operations())


class ExposeAcquaintanceGates(ExpandComposite):
    def __init__(self):
        self.extension = Extensions()
        self.no_decomp = lambda op: (
                not op_acquaintance_size(op) or
                (isinstance(op, GateOperation) and
                 isinstance(op.gate, AcquaintanceOpportunityGate)))

expose_acquaintance_gates = ExposeAcquaintanceGates()


def complete_acquaintance_strategy(qubit_order: Sequence[QubitId],
                                   acquaintance_size: int=0,
                                   ) -> AcquaintanceStrategy:
    if acquaintance_size < 0:
        raise ValueError('acquaintance_size must be non-negative.')
    elif acquaintance_size == 0:
        return AcquaintanceStrategy()

    strategy = AcquaintanceStrategy(
            Circuit.from_ops(ACQUAINT(q) for q in qubit_order))
    no_decomp = lambda op: (
            isinstance(op.gate,
                (CircularShiftGate, LinearPermutationGate)) or
            (isinstance(op.gate, SwapNetworkGate) and
                not op.gate.acquaintance_size))
    expand = ExpandComposite(no_decomp=no_decomp)
    for size_to_acquaint in range(2, acquaintance_size + 1):
        expand(strategy)
        strategy.nest(qubit_order, size_to_acquaint)
    return strategy
