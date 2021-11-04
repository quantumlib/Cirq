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

import numpy as np

import cirq


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
        self.gateset = cirq.Gateset(
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

    def qubit_set(self) -> AbstractSet['cirq.Qid']:
        return self.qubits

    def validate_operation(self, operation: cirq.Operation):
        if operation.gate is None:
            raise ValueError(
                f'IonQAPIDevice does not support operations with no gates {operation}.'
            )
        if not self.is_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')
        if not set(operation.qubits).intersection(self.qubit_set()):
            raise ValueError(f'Operation with qubits not on the device. Qubits: {operation.qubits}')

    def is_api_gate(self, operation: cirq.Operation) -> bool:
        return operation in self.gateset

    def decompose_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        if self.is_api_gate(operation):
            return operation
        assert cirq.has_unitary(operation), (
            f'Operation {operation} that is not available on the IonQ API nor does it have a '
            'unitary matrix to use to decompose it to the API.'
        )
        num_qubits = len(operation.qubits)
        if num_qubits == 1:
            return self._decompose_single_qubit(operation)
        if num_qubits == 2:
            return self._decompose_two_qubit(operation)
        raise ValueError(f'Operation {operation} not supported by IonQ API.')

    def _decompose_single_qubit(self, operation: cirq.Operation) -> cirq.OP_TREE:
        qubit = operation.qubits[0]
        mat = cirq.unitary(operation)
        for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    def _decompose_two_qubit(self, operation: cirq.Operation) -> cirq.OP_TREE:
        """Decomposes a two qubit gate into XXPow, YYPow, and ZZPow plus single qubit gates."""
        mat = cirq.unitary(operation)
        kak = cirq.kak_decomposition(mat, check_preconditions=False)

        for qubit, mat in zip(operation.qubits, kak.single_qubit_operations_before):
            gates = cirq.single_qubit_matrix_to_gates(mat, self.atol)
            for gate in gates:
                yield gate(qubit)

        two_qubit_gates = [cirq.XX, cirq.YY, cirq.ZZ]
        for two_qubit_gate, coefficient in zip(two_qubit_gates, kak.interaction_coefficients):
            yield (two_qubit_gate ** (-coefficient * 2 / np.pi))(*operation.qubits)

        for qubit, mat in zip(operation.qubits, kak.single_qubit_operations_after):
            for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
                yield gate(qubit)
