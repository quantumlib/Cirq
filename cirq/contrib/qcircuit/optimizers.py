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

import itertools

from typing import Callable, Optional

from cirq import circuits, ops


class PadBetweenOps(circuits.OptimizationPass):
    def __init__(self,
                 needs_padding: Callable[[ops.Operation,
                                          Optional[ops.Operation]], bool],
                 ) -> None:
        self.needs_padding = needs_padding

    def optimize_circuit(self, circuit: circuits.Circuit):
        for i in reversed(range(len(circuit) - 1)):
            op_pairs = itertools.product(circuit[i], circuit[i + 1])
            if any(self.needs_padding(*op_pair) for op_pair in op_pairs):
                circuit.insert(i + 1, ops.Moment())
        if any(self.needs_padding(op, None) for op in circuit[-1]):
            circuit.append(ops.Moment())

def swap_followed_by_non_swap(
        first_op: ops.Operation, second_op: Optional[ops.Operation]) -> bool:
    return bool(
        ((second_op is None) or
         (set(first_op.qubits) & set(second_op.qubits))) and
        isinstance(first_op, ops.GateOperation) and
        first_op.gate == ops.SWAP and
        not (isinstance(second_op, ops.GateOperation) and
             second_op.gate == ops.SWAP))

PadAfterSwapGates = PadBetweenOps(swap_followed_by_non_swap)

default_optimizers = (PadAfterSwapGates,)
