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
from typing import Sequence

from cirq.circuits import Circuit, Moment, ExpandComposite, InsertStrategy
from cirq.ops import QubitId, OP_TREE
from cirq.contrib.acquaintance.gates import (
     SwapNetworkGate, AcquaintanceOpportunityGate, ACQUAINT)
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import PermutationGate

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
        super().__init__(circuit.moments)

    def rectify(self):
        last_gate_type = self.gate_types[0]
        rectified_moments = []
        for moment in self:
            gate_type_to_ops = defaultdict(list)
            for op in moment.operations:
                gate_type_to_ops[type(op.gate)].append(op)
            if len(gate_type_to_ops) == 1:
                rectified_moments.append(moment)
                continue
            for gate_type in sorted(gate_type_to_ops, 
                                    key=partial(ne, last_gate_type)):
                rectified_moments.append(Moment(gate_type_to_ops[gate_type]))
                last_gate_type = gate_type
        self.moments = rectified_moments

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
                self.moments[moment_index] = Moment([swap_network_op])
                reflected = not reflected
        return reflected

    @staticmethod
    def from_ops(*operations: OP_TREE,
                 strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE
                 ) -> 'AcquaintanceStrategy':
        return AcquaintanceStrategy(Circuit.from_ops(*operations))

    def acquaintance_size(self) -> int:
        return max(len(op.qubits) for op in self.all_operations()
                   if isinstance(op.gate, AcquaintanceOpportunityGate))


def complete_acquaintance_strategy(qubit_order: Sequence[QubitId],
                                   acquaintance_size: int=0,
                                   ) -> AcquaintanceStrategy:
    if acquaintance_size < 0:
        raise ValueError('acquaintance_size must be non-negative.')
    elif acquaintance_size == 0:
        return AcquaintanceStrategy()

    strategy = AcquaintanceStrategy.from_ops(ACQUAINT(q) for q in qubit_order)
    is_shift = lambda op: isinstance(op.gate, CircularShiftGate)
    expand = ExpandComposite(no_decomp=is_shift)
    for size_to_acquaint in range(2, acquaintance_size + 1):
        expand(strategy)
        strategy.nest(qubit_order, size_to_acquaint)
    return strategy
