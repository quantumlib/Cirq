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

from typing import Dict, List, TYPE_CHECKING
from cirq import ops, value
if TYPE_CHECKING:
    import cirq


def defer_measurements(circuit: 'cirq.AbstractCircuit') -> 'cirq.Circuit':
    measurement_qubits: Dict[str, List['cirq.Qid']] = {}

    def defer(op: 'cirq.Operation') -> 'cirq.OP_TREE':
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            targets = [ops.NamedQid(f'{gate.key}-{q}', q.dimension) for q in op.qubits]
            measurement_qubits[gate.key] = targets
            cxs = [ops.CX(q, target) for q, target in zip(op.qubits, targets)]
            return cxs + [ops.X(targets[i]) for i, b in enumerate(gate.invert_mask) if b]
        elif isinstance(op, ops.ClassicallyControlledOperation):
            controls = []
            for c in op.classical_controls:
                if isinstance(c, value.KeyCondition):
                    controls.extend(measurement_qubits[str(c.key)])
                else:
                    raise ValueError('Only KeyConditions are allowed.')
            return ops.ControlledOperation(
                controls=controls, sub_operation=op.without_classical_controls()
            )
        return op

    circuit = circuit.map_operations(defer).unfreeze()
    for k, qubits in measurement_qubits.items():
        circuit.append(ops.measure(*qubits, key=k))
    return circuit
