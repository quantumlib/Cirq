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
"""Support for serializing gates supported by IonQ's API."""
from typing import cast, TYPE_CHECKING

import numpy as np

from cirq.ops import common_gates, gate_operation, parity_gates
from cirq.devices import line_qubit

if TYPE_CHECKING:
    import cirq


class Serializer:
    """Takes gates supported by IonQ's API and converts them to IonQ json form.

    Note that this does only serialization, it does not do any decomposition
    into the supported gate set.
    """

    def serialize(self, circuit: 'cirq.Circuit') -> dict:
        """Serialize the given circuit.

        Raises:
            ValueError: if the circuit has gates that are not supported or
                is otherwise invalid.
        """
        if len(circuit) == 0:
            raise ValueError('Cannot serialize empty circuit.')
        all_qubits = circuit.all_qubits()
        if any(not isinstance(q, line_qubit.LineQubit) for q in all_qubits):
            raise ValueError('All qubits must be cirq.LineQubits but were '
                             f'{set(type(q) for q in all_qubits)}')
        num_qubits = cast(line_qubit.LineQubit, max(all_qubits)).x + 1
        return {
            'qubits': num_qubits,
            'circuit': self._serialize_circuit(circuit, num_qubits)
        }

    def _serialize_circuit(self, circuit: 'cirq.Circuit',
                           num_qubits: int) -> list:
        return [self._serialize_op(op) for moment in circuit for op in moment]

    def _serialize_op(self, op: 'cirq.Operation') -> dict:
        if not isinstance(op, gate_operation.GateOperation):
            raise ValueError(
                'Attempt to serialize circuit with an operation which is '
                f'not a cirq.GateOperation. Type: {type(op)} Op: {op}.')
        gate_op = cast(gate_operation.GateOperation, op)
        targets = [cast(line_qubit.LineQubit, q).x for q in gate_op.qubits]
        if any(x < 0 for x in targets):
            raise ValueError(
                'IonQ API must use LineQubits from 0 to number of qubits - 1. '
                f'Instead found line qubits with indices {targets}.')
        gate = gate_op.gate
        if isinstance(gate, common_gates.XPowGate):
            # TODO: handle cases where this is a pauli x, and similar below
            # https://github.com/quantumlib/Cirq/issues/3479
            return {
                'gate': 'rx',
                'targets': targets,
                'rotation': gate.exponent * np.pi
            }
        elif isinstance(gate, common_gates.YPowGate):
            return {
                'gate': 'ry',
                'targets': targets,
                'rotation': gate.exponent * np.pi
            }
        elif isinstance(gate, common_gates.ZPowGate):
            return {
                'gate': 'rz',
                'targets': targets,
                'rotation': gate.exponent * np.pi
            }
        elif isinstance(gate, parity_gates.XXPowGate):
            return {
                'gate': 'xx',
                'targets': targets,
                'rotation': gate.exponent * np.pi
            }
        else:
            # Add complete set of serializable gates.
            # https://github.com/quantumlib/Cirq/issues/3479
            raise ValueError(
                f'Gate of type {type(gate)} is not serializable on the IonQ '
                f'API. Op: {op}.')
