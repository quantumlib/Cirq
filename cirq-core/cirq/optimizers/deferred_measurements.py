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

from typing import Any, Dict, List, Set, TYPE_CHECKING, TypeVar, Union

from cirq import circuits, ops, value

if TYPE_CHECKING:
    import cirq


class _MeasurementQid(ops.Qid):
    def __init__(self, key: Union[str, 'cirq.MeasurementKey'], qid: 'cirq.Qid'):
        self._key = value.MeasurementKey.parse_serialized(key) if isinstance(key, str) else key
        self._qid = qid

    @property
    def dimension(self) -> int:
        return self._qid.dimension

    def _comparison_key(self) -> Any:
        return (str(self._key), self._qid._comparison_key())

    def __str__(self) -> str:
        return f'{self._key} (q {self._qid})'  # coverage: ignore

    def __repr__(self) -> str:
        return f'_MeasurementQid({self._key!r}, {self._qid!r})'  # coverage: ignore


def defer_measurements(circuit: 'cirq.AbstractCircuit') -> 'cirq.Circuit':
    circuit = circuits.CircuitOperation(circuit.freeze()).mapped_circuit(deep=True)
    qubits_found: Set['cirq.Qid'] = set()
    terminal_measurements: Set['cirq.MeasurementKey'] = set()
    control_keys: Set['cirq.MeasurementKey'] = set()
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
    measurement_qubits: Dict['cirq.MeasurementKey', List['_MeasurementQid']] = {}

    def defer(op: 'cirq.Operation') -> 'cirq.OP_TREE':
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            if key in terminal_measurements:
                return op
            targets = [_MeasurementQid(key, q) for q in op.qubits]
            measurement_qubits[key] = targets
            cxs = [ops.CX(q, target) for q, target in zip(op.qubits, targets)]
            return cxs + [ops.X(targets[i]) for i, b in enumerate(gate.invert_mask) if b]
        elif isinstance(op, ops.ClassicallyControlledOperation):
            controls = []
            for c in op.classical_controls:
                if isinstance(c, value.KeyCondition):
                    qubits = measurement_qubits[c.key]
                    if len(qubits) != 1:
                        # Multi-qubit conditions require
                        # https://github.com/quantumlib/Cirq/issues/4512
                        raise ValueError('Only single qubit conditions are allowed.')
                    controls.extend(qubits)
                else:
                    raise ValueError('Only KeyConditions are allowed.')
            return ops.ControlledOperation(
                controls=controls,
                sub_operation=op.without_classical_controls(),
                control_values=[tuple(range(1, q.dimension)) for q in controls],
            )
        return op

    circuit = circuit.map_operations(defer).unfreeze()
    for k, qubits in measurement_qubits.items():
        circuit.append(ops.measure(*qubits, key=k))
    return circuit


CIRCUIT_TYPE = TypeVar('CIRCUIT_TYPE', bound='cirq.AbstractCircuit')


def dephase_measurements(circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
    def dephase(op: 'cirq.Operation') -> 'cirq.OP_TREE':
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            return ops.KrausChannel.from_channel(ops.phase_damp(1), key=key).on_each(op.qubits)
        elif isinstance(op, ops.ClassicallyControlledOperation):
            raise ValueError('Use cirq.defer_measurements first to remove classical controls.')
        elif isinstance(op, circuits.CircuitOperation):
            circuit = dephase_measurements(op.circuit)
            return op.replace(circuit=circuit)
        return op

    return circuit.map_operations(dephase)
