# Copyright 2020 The Cirq Developers
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

from cirq import ops, iqm, optimizers, protocols


def _convert_one(op: ops.Operation) -> ops.OP_TREE:
    matrix = protocols.unitary(op, None) if len(op.qubits) <= 2 else None
    if matrix is not None and len(op.qubits) == 1:
        return op  # TODO
    if matrix is not None and len(op.qubits) == 2:
        return optimizers.two_qubit_matrix_to_operations(
            op.qubits[0], op.qubits[1], matrix, allow_partial_czs=False)

    return NotImplemented


def convert(op: ops.Operation) -> ops.OP_TREE:
    """Attempts to convert a single (one- or two-qubit) operation into gates
    supported on IQM's Adonis device.
    """
    if _is_native_adonis_op(op):
        return op

    return protocols.decompose(op,
                               keep=_is_native_adonis_op,
                               intercepting_decomposer=_convert_one,
                               on_stuck_raise=None)


def _is_native_adonis_op(operation: ops.Operation):
    return isinstance(operation, ops.GateOperation) and isinstance(
        operation.gate, iqm.Adonis.SUPPORTED_GATES)
