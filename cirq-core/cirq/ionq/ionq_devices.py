# Copyright 2021 The Cirq Developers
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

from typing import AbstractSet, Callable, Dict, Sequence, Type, Union, TYPE_CHECKING

import numpy as np

from cirq import devices, linalg, ops, optimizers, protocols
from cirq.ops import common_gates, parity_gates

if TYPE_CHECKING:
    import cirq


class IonQAPIDevice(devices.Device):
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

    def __init__(self, qubits: Union[Sequence[devices.LineQubit], int], atol=1e-8):
        """Construct the device.

        Args:
            qubits: The qubits upon which this device acts or the number of qubits. If the number
                of qubits, then the qubits will be `cirq.LineQubit`s from 0 to this number minus
                one.
            atol: The absolute tolerance used for gate calculations and decompositions.
        """
        if isinstance(qubits, int):
            self.qubits = frozenset(devices.LineQubit.range(qubits))
        else:
            self.qubits = frozenset(qubits)
        self.atol = atol
        all_gates_valid = lambda x: True
        near_1_mod_2 = lambda x: abs(x.gate.exponent % 2 - 1) < self.atol
        self._is_api_gate_dispatch: Dict[Type['cirq.Gate'], Callable] = {
            common_gates.XPowGate: all_gates_valid,
            common_gates.YPowGate: all_gates_valid,
            common_gates.ZPowGate: all_gates_valid,
            parity_gates.XXPowGate: all_gates_valid,
            parity_gates.YYPowGate: all_gates_valid,
            parity_gates.ZZPowGate: all_gates_valid,
            common_gates.CNotPowGate: near_1_mod_2,
            common_gates.HPowGate: near_1_mod_2,
            common_gates.SwapPowGate: near_1_mod_2,
            common_gates.MeasurementGate: all_gates_valid,
        }

    def qubit_set(self) -> AbstractSet['cirq.Qid']:
        return self.qubits

    def validate_operation(self, operation: 'cirq.Operation'):
        if operation.gate is None:
            raise ValueError(
                f'IonQAPIDevice does not support operations with no gates {operation}.'
            )
        if not self.is_api_gate(operation):
            raise ValueError(f'IonQAPIDevice has unsupported gate {operation.gate}.')
        if not set(operation.qubits).intersection(self.qubit_set()):
            raise ValueError(f'Operation with qubits not on the device. Qubits: {operation.qubits}')

    def is_api_gate(self, operation: 'cirq.Operation') -> bool:
        gate = operation.gate
        for gate_mro_type in type(gate).mro():
            if gate_mro_type in self._is_api_gate_dispatch:
                return self._is_api_gate_dispatch[gate_mro_type](operation)
        return False

    def decompose_operation(self, operation: 'cirq.Operation') -> ops.OP_TREE:
        if self.is_api_gate(operation):
            return operation
        assert protocols.has_unitary(operation), (
            f'Operation {operation} that is not available on the IonQ API nor does it have a '
            'unitary matrix to use to decompose it to the API.'
        )
        num_qubits = len(operation.qubits)
        if num_qubits == 1:
            return self._decompose_single_qubit(operation)
        if num_qubits == 2:
            return self._decompose_two_qubit(operation)
        raise ValueError('Operation {operation} not supported by IonQ API.')

    def _decompose_single_qubit(self, operation: 'cirq.Operation') -> ops.OP_TREE:
        qubit = operation.qubits[0]
        mat = protocols.unitary(operation)
        for gate in optimizers.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    def _decompose_two_qubit(self, operation: 'cirq.Operation') -> ops.OP_TREE:
        """Decomposes a two qubit gate into XXPow, YYPow, and ZZPow plus single qubit gates."""
        mat = protocols.unitary(operation)
        kak = linalg.kak_decomposition(mat, check_preconditions=False)

        for qubit, mat in zip(operation.qubits, kak.single_qubit_operations_before):
            gates = optimizers.single_qubit_matrix_to_gates(mat, self.atol)
            for gate in gates:
                yield gate(qubit)

        two_qubit_gates = [parity_gates.XX, parity_gates.YY, parity_gates.ZZ]
        for two_qubit_gate, coefficient in zip(two_qubit_gates, kak.interaction_coefficients):
            yield (two_qubit_gate ** (-coefficient * 2 / np.pi))(*operation.qubits)

        for qubit, mat in zip(operation.qubits, kak.single_qubit_operations_after):
            for gate in optimizers.single_qubit_matrix_to_gates(mat, self.atol):
                yield gate(qubit)
