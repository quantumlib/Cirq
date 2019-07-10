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


def ensure_final_moment_empty(circuit: circuits.Circuit) -> None:
    """Ensures that the final moment is empty, by added an empty one if not."""
    if circuit[-1]:
        circuit.append(ops.Moment())


def pad_between_ops(
        padding_needed: Callable[[ops.Operation, Optional[ops.Operation]], int]
) -> Callable[[circuits.Circuit], None]:
    """Inserts padding where needed as indicated by the specified function."""

    def optimize_circuit(circuit):
        for i in reversed(range(len(circuit) - 1)):
            op_pairs = itertools.product(circuit[i], circuit[i + 1])
            padding = max(
                itertools.chain(
                    (0,), (padding_needed(*op_pair) for op_pair in op_pairs)))
            circuit.insert(i + 1, (ops.Moment(),) * padding)
        padding = max(
            itertools.chain((0,),
                            (padding_needed(op, None) for op in circuit[-1])))
        circuit.append((ops.Moment(),) * padding)

    return optimize_circuit


def swap_followed_by_non_swap(first_op: ops.Operation,
                              second_op: Optional[ops.Operation]) -> int:
    """Returns the amount of padding needed between two operations in a
    qcircuit diagram.

    If a swap is followed by a non-swap gate (on an overlapping set of qubits),
    1 layer of padding is needed.

    If a swap is followed by nothing (i.e., it's the end of the circuit), 2
    layers are needed.

    Otherwise, no padding is needed."""

    if not (isinstance(first_op, ops.GateOperation) and
            first_op.gate == ops.SWAP):
        return 0
    if second_op is None:
        return 2
    if not (set(first_op.qubits) & set(second_op.qubits)):
        return 0
    if (isinstance(second_op, ops.GateOperation) and
            second_op.gate == ops.SWAP):
        return 0
    return 1


pad_after_swap_gates = pad_between_ops(swap_followed_by_non_swap)

default_optimizers = (pad_after_swap_gates, ensure_final_moment_empty)
