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

from typing import Any, Dict, List, Optional
import cirq
from cirq import protocols
from cirq.transformers import transformer_api, drop_empty_moments, drop_negligible_operations
from cirq.transformers.target_gatesets import compilation_target_gateset


@cirq.transformer
def merge_to_phased_x_and_z(
    c: cirq.Circuit, *, context: Optional['cirq.TransformerContext'] = None
) -> cirq.Circuit:
    return cirq.merge_single_qubit_gates_to_phased_x_and_z(c)


@cirq.transformer
def decompose_phased_x_pow(
    c: cirq.Circuit, *, context: Optional['cirq.TransformerContext'] = None
) -> cirq.Circuit:
    return cirq.map_operations_and_unroll(
        c, lambda op, _: cirq.decompose_once(op) if type(op.gate) == cirq.PhasedXPowGate else op
    )


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
    gateset = IonQCompilationTargetGateset(atol)
    if operation in gateset:
        return operation

    assert cirq.has_unitary(operation), (
        f'Operation {operation} is not available on the IonQ API nor does it have a '
        'unitary matrix to use to decompose it to the API.'
    )
    num_qubits = len(operation.qubits)
    if num_qubits <= 2:
        return list(
            cirq.optimize_for_target_gateset(
                cirq.Circuit(operation), gateset=IonQCompilationTargetGateset(atol)
            ).all_operations()
        )
    raise ValueError(f'Operation {operation} not supported by IonQ API.')


class IonQCompilationTargetGateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
    """A two-qubit target gateset for gates exposed by the IonQ API.

    When using this in constructing a circuit, it will convert one and two qubit gates
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

    def __init__(self, atol=1e-8):
        """Construct the gateset

        Args:
            atol: The absolute tolerance used for gate calculations and decompositions.
        """
        super().__init__(
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
            name='IonQCompilationTargetGateset',
        )
        self.atol = atol

    def __repr__(self) -> str:
        return f'cirq_ionq.IonQCompilationTargetGateset(atol={self.atol})'

    def is_api_gate(self, operation: cirq.Operation) -> bool:
        return operation in self

    def _decompose_single_qubit_operation(self, operation: cirq.Operation, _) -> 'cirq.OP_TREE':
        if self.is_api_gate(operation):
            return operation
        if not protocols.has_unitary(operation):
            raise ValueError(f'Operation {operation} not supported by IonQ API.')
        qubit = operation.qubits[0]
        mat = cirq.unitary(operation)
        return [gate(qubit) for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol)]

    def _decompose_two_qubit_operation(self, operation: cirq.Operation, _) -> 'cirq.OP_TREE':
        """Decomposes a two qubit unitary operation into ZPOW, XPOW, and CNOT."""
        if self.is_api_gate(operation):
            return operation
        if not protocols.has_unitary(operation):
            raise ValueError(f'Operation {operation} not supported by IonQ API.')
        mat = cirq.unitary(operation)
        q0, q1 = operation.qubits
        naive = cirq.two_qubit_matrix_to_cz_operations(
            q0, q1, mat, allow_partial_czs=False, atol=self.atol
        )
        return cirq.map_operations_and_unroll(
            cirq.Circuit(naive),
            lambda op, _: [cirq.H(op.qubits[1]), cirq.CNOT(*op.qubits), cirq.H(op.qubits[1])]
            if type(op.gate) == cirq.CZPowGate
            else op,
        )

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> Dict[str, Any]:
        return {'atol': self.atol}

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [
            merge_to_phased_x_and_z,
            decompose_phased_x_pow,
            drop_negligible_operations,
            drop_empty_moments,
        ]
