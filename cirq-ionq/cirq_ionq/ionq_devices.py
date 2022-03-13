# Copyright 2021 The Cirq Developers
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
"""Devices for IonQ hardware."""

from typing import AbstractSet, Sequence, Union

import cirq
from cirq import _compat


_VALID_GATES = cirq.Gateset(
    cirq.H,
    cirq.CNOT,
    cirq.SWAP,
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
    cirq.XXPowGate,
    cirq.YYPowGate,
    cirq.ZZPowGate,
    cirq.MeasurementGate,
    unroll_circuit_op=False,
    accept_global_phase_op=False,
)


class IonQAPIDevice(cirq.Device):
    """A device that uses the gates exposed by the IonQ API.

    When using this device in constructing a circuit, it will convert one and two qubit gates
    that are not supported by the API into those supported by the API if they have a unitary
    matrix (support the unitary protocol).

    Note that this device does not do any compression of the resulting circuit, i.e. it may
    result in a series of single qubit gates that could be executed using far fewer elements.

    The gates supported by the API are
        * `cirq.XPowGate`, `cirq.YPowGate`, `cirq.ZPowGate`
        * `cirq.XXPowGate`, `cirq.YYPowGate`, `cirq.ZZPowGate`
        * `cirq.CNOT`, `cirq.H`, `cirq.SWAP`
        * `cirq.MeasurementGate`
    """

    def __init__(self, qubits: Union[Sequence[cirq.LineQubit], int], atol=1e-8):
        """Construct the device.

        Args:
            qubits: The qubits upon which this device acts or the number of qubits. If the number
                of qubits, then the qubits will be `cirq.LineQubit`s from 0 to this number minus
                one.
            atol: The absolute tolerance used for gate calculations and decompositions.
        """
        if isinstance(qubits, int):
            self.qubits = frozenset(cirq.LineQubit.range(qubits))
        else:
            self.qubits = frozenset(qubits)
        self.atol = atol
        self._metadata = cirq.DeviceMetadata(
            self.qubits,
            [(a, b) for a in self.qubits for b in self.qubits if a != b],
        )

    @property
    def metadata(self) -> cirq.DeviceMetadata:
        return self._metadata

    @_compat.deprecated(
        fix='Use metadata.qubit_set if applicable.',
        deadline='v0.15',
    )
    def qubit_set(self) -> AbstractSet['cirq.Qid']:
        return self.qubits

    def validate_operation(self, operation: cirq.Operation):
        if operation.gate is None:
            raise ValueError(
                f'IonQAPIDevice does not support operations with no gates {operation}.'
            )
        if not self.is_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')
        if not set(operation.qubits).intersection(self.metadata.qubit_set):
            raise ValueError(f'Operation with qubits not on the device. Qubits: {operation.qubits}')

    def is_api_gate(self, operation: cirq.Operation) -> bool:
        return operation in _VALID_GATES

    @_compat.deprecated(
        fix='Use cirq_ionq.decompose_to_device operation instead.',
        deadline='v0.15',
    )
    def decompose_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        return decompose_to_device(operation)


def decompose_to_device(operation: cirq.Operation, atol: float = 1e-8) -> cirq.OP_TREE:
    """Decompose operation to ionq native operations.


    Merges single qubit operations and decomposes two qubit operations
    into CZ gates.

    Args:
        operation: `cirq.Operation` to decompose.
        atol: absolute error tolerance to use when declaring two unitary
            operations equal.

    Returns:
        cirq.OP_TREE containing decomposed operations.

    Raises:
        ValueError: If supplied operation cannot be decomposed
            for the ionq device.

    """
    if operation in _VALID_GATES:
        return operation
    assert cirq.has_unitary(operation), (
        f'Operation {operation} is not available on the IonQ API nor does it have a '
        'unitary matrix to use to decompose it to the API.'
    )
    num_qubits = len(operation.qubits)
    if num_qubits == 1:
        return _decompose_single_qubit(operation, atol)
    if num_qubits == 2:
        return _decompose_two_qubit(operation)
    raise ValueError(f'Operation {operation} not supported by IonQ API.')


def _decompose_single_qubit(operation: cirq.Operation, atol: float) -> cirq.OP_TREE:
    qubit = operation.qubits[0]
    mat = cirq.unitary(operation)
    for gate in cirq.single_qubit_matrix_to_gates(mat, atol):
        yield gate(qubit)


def _decompose_two_qubit(operation: cirq.Operation) -> cirq.OP_TREE:
    """Decomposes a two qubit unitary operation into ZPOW, XPOW, and CNOT."""
    mat = cirq.unitary(operation)
    q0, q1 = operation.qubits
    naive = cirq.two_qubit_matrix_to_cz_operations(q0, q1, mat, allow_partial_czs=False)
    temp = cirq.map_operations_and_unroll(
        cirq.Circuit(naive),
        lambda op, _: [cirq.H(op.qubits[1]), cirq.CNOT(*op.qubits), cirq.H(op.qubits[1])]
        if type(op.gate) == cirq.CZPowGate
        else op,
    )
    temp = cirq.merge_single_qubit_gates_to_phased_x_and_z(temp)
    # A final pass breaks up PhasedXPow into Rz, Rx.
    yield cirq.map_operations_and_unroll(
        temp,
        lambda op, _: cirq.decompose_once(op) if type(op.gate) == cirq.PhasedXPowGate else op,
    ).all_operations()
