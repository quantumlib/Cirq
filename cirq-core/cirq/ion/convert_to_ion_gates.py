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

import numpy as np

from cirq import ops, protocols, optimizers, circuits, transformers
from cirq.ion import ms, two_qubit_matrix_to_ion_operations, ion_device


class ConvertToIonGates:
    """Attempts to convert non-native gates into IonGates."""

    def __init__(self, ignore_failures: bool = False) -> None:
        """Inits ConvertToIonGates.

        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
        """
        super().__init__()
        self.ignore_failures = ignore_failures
        self.gateset = ion_device.get_ion_gateset()

    def convert_one(self, op: ops.Operation) -> ops.OP_TREE:
        """Convert a single (one- or two-qubit) operation into ion trap native gates.

        Args:
            op: The gate operation to be converted.

        Returns:
            The desired operations implemented with ion trap gates.

        Raises:
            TypeError: If the operation cannot be converted.
        """

        # Known gate name
        if not isinstance(op, ops.GateOperation):
            raise TypeError(f"{op!r} is not a gate operation.")

        if op in self.gateset:
            return [op]
        # one choice of known Hadamard gate decomposition
        if isinstance(op.gate, ops.HPowGate) and op.gate.exponent == 1:
            return [ops.rx(np.pi).on(op.qubits[0]), ops.ry(-1 * np.pi / 2).on(op.qubits[0])]
        # one choice of known CNOT gate decomposition
        if isinstance(op.gate, ops.CNotPowGate) and op.gate.exponent == 1:
            return [
                ops.ry(np.pi / 2).on(op.qubits[0]),
                ms(np.pi / 4).on(op.qubits[0], op.qubits[1]),
                ops.rx(-1 * np.pi / 2).on(op.qubits[0]),
                ops.rx(-1 * np.pi / 2).on(op.qubits[1]),
                ops.ry(-1 * np.pi / 2).on(op.qubits[0]),
            ]
        # Known matrix
        mat = protocols.unitary(op, None) if len(op.qubits) <= 2 else None
        if mat is not None and len(op.qubits) == 1:
            gates = transformers.single_qubit_matrix_to_phased_x_z(mat)
            return [g.on(op.qubits[0]) for g in gates]
        if mat is not None and len(op.qubits) == 2:
            return two_qubit_matrix_to_ion_operations(op.qubits[0], op.qubits[1], mat)

        if self.ignore_failures:
            return [op]

        raise TypeError(
            "Don't know how to work with {!r}. "
            "It isn't a native Ion Trap operation, "
            "a 1 or 2 qubit gate with a known unitary, "
            "or composite.".format(op.gate)
        )

    def convert_circuit(self, circuit: circuits.Circuit) -> circuits.Circuit:
        new_circuit = circuits.Circuit()
        for moment in circuit:
            for op in moment.operations:
                new_circuit.append(self.convert_one(op))
        optimizers.merge_single_qubit_gates_into_phased_x_z(new_circuit)

        return new_circuit
