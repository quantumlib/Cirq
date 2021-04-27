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
import dataclasses

from typing import (
    Callable,
    cast,
    Collection,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Type,
    TYPE_CHECKING,
)

import numpy as np

from cirq import protocols
from cirq.ops import common_gates, parity_gates
from cirq.devices import line_qubit

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class SerializedProgram:
    """A container for the serialized portions of a `cirq.Circuit`.

    Attributes:
        body: A dictionary which containts the number of qubits and the serialized circuit
            minus the measurements.
        metadata: A dictionary whose keys store information about the measurements in the circuit.
    """

    body: dict
    metadata: dict


class Serializer:
    """Takes gates supported by IonQ's API and converts them to IonQ json form.

    Note that this does only serialization, it does not do any decomposition into the supported
    gate set.
    """

    def __init__(self, atol: float = 1e-8):
        """Create the Serializer.

        Args:
            atol: Absolute tolerance used in determining whether a gate with a float parameter
                should be serialized as a gate rounded to that parameter. Defaults to 1e-8.
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
            common_gates.MeasurementGate: self._serialize_measurement_gate,
        }

    def serialize(self, circuit: 'cirq.Circuit') -> SerializedProgram:
        """Serialize the given circuit.

        Raises:
            ValueError: if the circuit has gates that are not supported or is otherwise invalid.
        """
        self._validate_circuit(circuit)
        num_qubits = self._validate_qubits(circuit.all_qubits())

        serialized_ops = self._serialize_circuit(circuit)

        # IonQ API does not support measurements, so we pass the measurement keys through
        # the metadata field.  Here we split these out of the serialized ops.
        body = {
            'qubits': num_qubits,
            'circuit': [op for op in serialized_ops if op['gate'] != 'meas'],
        }
        metadata = self._serialize_measurements(op for op in serialized_ops if op['gate'] == 'meas')
        return SerializedProgram(body=body, metadata=metadata)

    def _validate_circuit(self, circuit: 'cirq.Circuit'):
        if len(circuit) == 0:
            raise ValueError('Cannot serialize empty circuit.')
        if not circuit.are_all_measurements_terminal():
            raise ValueError('All measurements in circuit must be at end of circuit.')

    def _validate_qubits(self, all_qubits: Collection['cirq.Qid']) -> int:
        """Returns the number of qubits while validating qubit types and values."""
        if any(not isinstance(q, line_qubit.LineQubit) for q in all_qubits):
            raise ValueError(
                f'All qubits must be cirq.LineQubits but were {set(type(q) for q in all_qubits)}'
            )
        if any(cast(line_qubit.LineQubit, q).x < 0 for q in all_qubits):
            raise ValueError(
                'IonQ API must use LineQubits from 0 to number of qubits - 1. Instead found line '
                f'qubits with indices {all_qubits}.'
            )
        num_qubits = cast(line_qubit.LineQubit, max(all_qubits)).x + 1
        return num_qubits

    def _serialize_circuit(self, circuit: 'cirq.Circuit') -> list:
        return [self._serialize_op(op) for moment in circuit for op in moment]

    def _serialize_op(self, op: 'cirq.Operation') -> dict:
        if op.gate is None:
            raise ValueError(
                'Attempt to serialize circuit with an operation which does not have a gate. '
                f'Type: {type(op)} Op: {op}.'
            )
        targets = [cast(line_qubit.LineQubit, q).x for q in op.qubits]
        gate = op.gate
        if protocols.is_parameterized(gate):
            raise ValueError(
                f'IonQ API does not support parameterized gates. Gate {gate} was parameterized. '
                'Consider resolving before sending.'
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

    def _serialize_measurement_gate(
        self, gate: 'cirq.MeasurementGate', targets: Sequence[int]
    ) -> dict:
        key = protocols.measurement_key(gate)
        if chr(31) in key or chr(30) in key:
            raise ValueError(
                'Measurement gates for IonQ API cannot have a key with a ascii unit'
                f'or record separator in it. Key was {key}'
            )
        return {'gate': 'meas', 'key': key, 'targets': ','.join(str(t) for t in targets)}

    def _near_mod_n(self, e: float, t: float, n: float) -> bool:
        """Returns whether a value, e, translated by t, is equal to 0 mod n."""
        return abs((e - t + 1) % n - 1) <= self.atol

    def _serialize_measurements(self, meas_ops: Iterator) -> Dict[str, str]:
        """Serializes measurement ops into a form suitable to be passed via metadata.

        IonQ API does not contain measurement gates, so we serialize measurement gate keys
        and targets into a form that is suitable for passing through IonQ's metadata field
        for a job.

        Each key and targets are serialized into a string of the form `key` + the ASCII unit
        separator (chr(31)) + targets as a comma separated value.  These are then combined
        into a string with a seperator character of the ASCII record separator (chr(30)).
        Finally this full string is serialized as the values in the metadata dict with keys
        given by `measurementX` for X = 0,1, .. 9 and X large enough to contain the entire
        string.

        Args:
            A list of the result of serializing the measurement (not supported by the API).

        Returns:
            The metadata dict that can be passed to the API.

        Raises:
            ValueError: if the
        """
        key_values = [f'{op["key"]}{chr(31)}{op["targets"]}' for op in meas_ops]
        full_str = chr(30).join(key_values)
        # IonQ maximum value size for metadata.
        max_value_size = 40
        split_strs = [
            full_str[i : i + max_value_size] for i in range(0, len(full_str), max_value_size)
        ]
        if len(split_strs) > 9:
            raise ValueError(
                'Measurement keys plus target strings too long for IonQ API. Please use '
                'smaller keys.'
            )
        return {f'measurement{i}': x for i, x in enumerate(split_strs)}
