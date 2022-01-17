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

"""An optimization pass that aligns gates to the left of the circuit."""

from typing import Dict, List, TYPE_CHECKING, Any
from cirq import circuits, ops, value

if TYPE_CHECKING:
    import cirq


class MeasurementQid(ops.Qid):
    def __init__(self, key: 'cirq.MeasurementKey', qid: 'cirq.Qid'):
        self._key = key
        self._qid = qid

    @property
    def dimension(self) -> int:
        return self._qid.dimension

    def _comparison_key(self) -> Any:
        return (str(self._key), self._qid._comparison_key())

    def __str__(self) -> str:
        return f'{self._key} {self._qid}'


def defer_measurements(
    circuit: 'cirq.AbstractCircuit', dephase_measurements=False
) -> 'cirq.Circuit':
    circuit = circuits.CircuitOperation(circuit.freeze()).mapped_circuit(deep=True)
    qubits_found = set()
    terminal_measurements = set()
    control_keys = set()
    for op in reversed(list(circuit.all_operations())):
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            if key not in control_keys and not any(q in qubits_found for q in op.qubits):
                terminal_measurements.add(key)
        elif isinstance(op, ops.ClassicallyControlledOperation):
            for c in op.classical_controls:
                control_keys.update(c.keys)
        qubits_found.update(op.qubits)
    measurement_qubits: Dict['cirq.MeasurementKey', List['cirq.Qid']] = {}

    def dephase_if_needed(op: 'cirq.Operation'):
        gate = op.gate
        assert isinstance(gate, ops.MeasurementGate)
        return (
            op
            if not dephase_measurements
            else ops.KrausChannel.from_channel(ops.phase_damp(1), key=gate.key).on(*op.qubits)
        )

    def defer(op: 'cirq.Operation') -> 'cirq.OP_TREE':
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            print(key)
            if key in terminal_measurements:
                return dephase_if_needed(op)
            print(key)
            print(key)
            targets = [MeasurementQid(key, q) for q in op.qubits]
            measurement_qubits[key] = targets
            cxs = [ops.CX(q, target) for q, target in zip(op.qubits, targets)]
            return cxs + [ops.X(targets[i]) for i, b in enumerate(gate.invert_mask) if b]
        elif isinstance(op, ops.ClassicallyControlledOperation):
            controls = []
            for c in op.classical_controls:
                if isinstance(c, value.KeyCondition):
                    controls.extend(measurement_qubits[c.key])
                else:
                    raise ValueError('Only KeyConditions are allowed.')
            # Depends on issue #4512, as we need some way of defining the condition "at least one
            # qubit is not zero" to match the classical interpretation of a multi-qubit measurement
            return ops.ControlledOperation(
                controls=controls, sub_operation=op.without_classical_controls()
            )
        return op

    circuit = circuit.map_operations(defer).unfreeze()
    for k, qubits in measurement_qubits.items():
        circuit.append(dephase_if_needed(ops.measure(*qubits, key=k)))
    return circuit
