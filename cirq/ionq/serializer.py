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
from typing import Callable, cast, Dict, Optional, Sequence, Type, TYPE_CHECKING

import numpy as np

from cirq import protocols
from cirq.ops import common_gates, gate_operation, parity_gates
from cirq.devices import line_qubit

if TYPE_CHECKING:
    import cirq


class Serializer:
    """Takes gates supported by IonQ's API and converts them to IonQ json form.

    Note that this does only serialization, it does not do any decomposition
    into the supported gate set.
    """

    def __init__(self, atol: float = 1e-8):
        """Create the Serializer.

        Args:
            atol: Absolute tolerance used in determining whether a gate with
                a float parameter should be serialized as a gate rounded
                to that parameter. Defaults to 1e-8.
        """
        self.atol = atol
        self._dispatch: Dict[Type['cirq.Gate'], Callable] = {
            common_gates.XPowGate: self._serialize_x_pow_gate,
            common_gates.YPowGate: self._serialize_y_pow_gate,
            common_gates.ZPowGate: self._serialize_z_pow_gate,
            parity_gates.XXPowGate: self._serialize_xx_pow_gate,
            parity_gates.YYPowGate: self._serialize_yy_pow_gate,
            parity_gates.ZZPowGate: self._serialize_zz_pow_gate,
            common_gates.CNotPowGate: self._serialize_cnot_pow_gate,
            common_gates.HPowGate: self._serialize_h_pow_gate,
            common_gates.SwapPowGate: self._serialize_swap_gate,
        }

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
            raise ValueError(
                f'All qubits must be cirq.LineQubits but were {set(type(q) for q in all_qubits)}'
            )
        num_qubits = cast(line_qubit.LineQubit, max(all_qubits)).x + 1
        return {'qubits': num_qubits, 'circuit': self._serialize_circuit(circuit, num_qubits)}

    def _serialize_circuit(self, circuit: 'cirq.Circuit', num_qubits: int) -> list:
        return [self._serialize_op(op) for moment in circuit for op in moment]

    def _serialize_op(self, op: 'cirq.Operation') -> dict:
        if not isinstance(op, gate_operation.GateOperation):
            raise ValueError(
                'Attempt to serialize circuit with an operation which is '
                f'not a cirq.GateOperation. Type: {type(op)} Op: {op}.'
            )
        gate_op = cast(gate_operation.GateOperation, op)
        targets = [cast(line_qubit.LineQubit, q).x for q in gate_op.qubits]
        if any(x < 0 for x in targets):
            raise ValueError(
                'IonQ API must use LineQubits from 0 to number of qubits - 1. '
                f'Instead found line qubits with indices {targets}.'
            )
        gate = gate_op.gate
        if protocols.is_parameterized(gate):
            raise ValueError(
                f'IonQ API does not support parameterized gates. Gate {gate} '
                'was parameterized. Consider resolving before sending'
            )
        gate_type = type(gate)
        # Check all superclasses.
        for gate_mro_type in gate_type.mro():
            if gate_mro_type in self._dispatch:
                serialized_op = self._dispatch[gate_mro_type](gate, targets)
                if serialized_op:
                    return serialized_op
        raise ValueError(f'Gate {gate} acting on {targets} cannot be serialized by IonQ API.')

    def _serialize_x_pow_gate(self, gate: 'cirq.XPowGate', targets: Sequence[int]) -> dict:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {'gate': 'x', 'targets': targets}
        elif self._near_mod_n(gate.exponent, 0.5, 2):
            return {'gate': 'v', 'targets': targets}
        elif self._near_mod_n(gate.exponent, -0.5, 2):
            return {'gate': 'vi', 'targets': targets}
        return {'gate': 'rx', 'targets': targets, 'rotation': gate.exponent * np.pi}

    def _serialize_y_pow_gate(self, gate: 'cirq.YPowGate', targets: Sequence[int]) -> dict:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {
                'gate': 'y',
                'targets': targets,
            }
        return {'gate': 'ry', 'targets': targets, 'rotation': gate.exponent * np.pi}

    def _serialize_z_pow_gate(self, gate: 'cirq.ZPowGate', targets: Sequence[int]) -> dict:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {
                'gate': 'z',
                'targets': targets,
            }
        elif self._near_mod_n(gate.exponent, 0.5, 2):
            return {'gate': 's', 'targets': targets}
        elif self._near_mod_n(gate.exponent, -0.5, 2):
            return {'gate': 'si', 'targets': targets}
        elif self._near_mod_n(gate.exponent, 0.25, 2):
            return {
                'gate': 't',
                'targets': targets,
            }
        elif self._near_mod_n(gate.exponent, -0.25, 2):
            return {
                'gate': 'ti',
                'targets': targets,
            }
        return {'gate': 'rz', 'targets': targets, 'rotation': gate.exponent * np.pi}

    def _serialize_xx_pow_gate(self, gate: 'cirq.XXPowGate', targets: Sequence[int]) -> dict:
        return self._serialize_parity_pow_gate(gate, targets, 'xx')

    def _serialize_yy_pow_gate(self, gate: 'cirq.YYPowGate', targets: Sequence[int]) -> dict:
        return self._serialize_parity_pow_gate(gate, targets, 'yy')

    def _serialize_zz_pow_gate(self, gate: 'cirq.ZZPowGate', targets: Sequence[int]) -> dict:
        return self._serialize_parity_pow_gate(gate, targets, 'zz')

    def _serialize_parity_pow_gate(
        self, gate: 'cirq.EigenGate', targets: Sequence[int], name: str
    ) -> dict:
        return {'gate': name, 'targets': targets, 'rotation': gate.exponent * np.pi}

    def _serialize_swap_gate(
        self, gate: 'cirq.SwapPowGate', targets: Sequence[int]
    ) -> Optional[dict]:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {
                'gate': 'swap',
                'targets': targets,
            }
        return None

    def _serialize_h_pow_gate(
        self, gate: 'cirq.HPowGate', targets: Sequence[int]
    ) -> Optional[dict]:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {
                'gate': 'h',
                'targets': targets,
            }
        return None

    def _serialize_cnot_pow_gate(
        self, gate: 'cirq.CNotPowGate', targets: Sequence[int]
    ) -> Optional[dict]:
        if self._near_mod_n(gate.exponent, 1, 2):
            return {'gate': 'cnot', 'control': targets[0], 'target': targets[1]}
        return None

    def _near_mod_n(self, e: float, t: float, n: float) -> bool:
        """Returns whether a value, e, translated by t, is equal to 0 mod n."""
        return abs((e - t + 1) % n - 1) <= self.atol
