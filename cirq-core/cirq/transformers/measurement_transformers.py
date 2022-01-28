# Copyright 2022 The Cirq Developers
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

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, TypeVar, Union

from cirq import circuits, ops, protocols, value
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


class _MeasurementQid(ops.Qid):
    """A qubit that substitutes in for a deferred measurement.

    Exactly one qubit will be created per qubit in the measurement gate.
    """

    def __init__(self, key: Union[str, 'cirq.MeasurementKey'], qid: 'cirq.Qid'):
        """Initializes the qubit.

        Args:
            key: The key of the measurement gate being deferred.
            qid: One qubit that is being measured. Each deferred measurement
                should create one new _MeasurementQid per qubit being measured
                by that gate.
        """
        self._key = value.MeasurementKey.parse_serialized(key) if isinstance(key, str) else key
        self._qid = qid

    @property
    def dimension(self) -> int:
        return self._qid.dimension

    def _comparison_key(self) -> Any:
        return (str(self._key), self._qid._comparison_key())

    def __str__(self) -> str:
        return f"M('{self._key}', q={self._qid})"

    def __repr__(self) -> str:
        return f'_MeasurementQid({self._key!r}, {self._qid!r})'


@transformer_api.transformer
def defer_measurements(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> 'cirq.Circuit':
    """Implements the Deferred Measurement Principle.

    Uses the Deferred Measurement Principle to move all measurements to the
    end of the circuit. All non-terminal measurements are changed to
    conditional quantum gates onto ancilla qubits, and classically controlled
    operations are transformed to quantum controls from those ancilla qubits.
    Finally, measurements of all ancilla qubits are appended to the end of the
    circuit.

    Optimizing deferred measurements is an area of active research, and future
    iterations may contain optimizations that reduce the number of ancilla
    qubits, so one should not depend on the exact shape of the output from this
    function. Only the logical equivalence is guaranteed to remain unchanged.
    Moment and subcircuit structure is not preserved.

    Args:
        circuit: The circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options
            for transformers.
    Returns:
        A circuit with equivalent logic, but all measurements at the end of the
        circuit.
    Raises:
        ValueError: If sympy-based classical conditions are used, or if
            conditions based on multi-qubit measurements exist. (The latter of
            these is planned to be implemented soon).
    """

    circuit = transformer_primitives.unroll_circuit_op(circuit, deep=True, tags_to_check=None)
    qubits_found: Set['cirq.Qid'] = set()
    terminal_measurements: Set['cirq.MeasurementKey'] = set()
    control_keys: Set['cirq.MeasurementKey'] = set()
    for op in reversed(list(circuit.all_operations())):
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            if key not in control_keys and qubits_found.isdisjoint(op.qubits):
                terminal_measurements.add(key)
        elif isinstance(op, ops.ClassicallyControlledOperation):
            for c in op.classical_controls:
                control_keys.update(c.keys)
        qubits_found.update(op.qubits)
    measurement_qubits: Dict['cirq.MeasurementKey', List['_MeasurementQid']] = {}

    def defer(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            key = value.MeasurementKey.parse_serialized(gate.key)
            if key in terminal_measurements:
                return op
            targets = [_MeasurementQid(key, q) for q in op.qubits]
            measurement_qubits[key] = targets
            cxs = [ops.CX(q, target) for q, target in zip(op.qubits, targets)]
            xs = [ops.X(targets[i]) for i, b in enumerate(gate.invert_mask) if b]  # type: ignore
            return cxs + xs
        elif protocols.is_measurement(op):
            return [defer(op, None) for op in protocols.decompose_once(op)]
        elif isinstance(op, ops.ClassicallyControlledOperation):
            controls = []
            for c in op.classical_controls:
                if isinstance(c, value.KeyCondition):
                    qubits = measurement_qubits[c.key]
                    if len(qubits) != 1:
                        # TODO: Multi-qubit conditions require
                        # https://github.com/quantumlib/Cirq/issues/4512
                        # Remember to update docstring above once this works.
                        raise ValueError('Only single qubit conditions are allowed.')
                    controls.extend(qubits)
                else:
                    raise ValueError('Only KeyConditions are allowed.')
            return op.without_classical_controls().controlled_by(
                *controls, control_values=[tuple(range(1, q.dimension)) for q in controls]
            )
        return op

    circuit = transformer_primitives.map_operations(
        circuit, defer, raise_if_add_qubits=False
    ).unfreeze()
    for k, qubits in measurement_qubits.items():
        circuit.append(ops.measure(*qubits, key=k))
    return circuit


CIRCUIT_TYPE = TypeVar('CIRCUIT_TYPE', bound='cirq.AbstractCircuit')


@transformer_api.transformer
def dephase_measurements(
    circuit: CIRCUIT_TYPE, *, context: Optional['cirq.TransformerContext'] = None
) -> CIRCUIT_TYPE:
    """Changes all measurements to a dephase operation.

    This transformer is useful when using a density matrix simulator, when
    wishing to calculate the final density matrix of a circuit and not simulate
    the measurements themselves.

    Args:
        circuit: The circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options
            for transformers.
    Returns:
        A copy of the circuit, with dephase operations in place of all
        measurements.
    Raises:
        ValueError: If the circuit contains classical controls. In this case,
            it is required to change these to quantum controls via
            `cirq.defer_measurements` first. Since deferral adds ancilla qubits
            to the circuit, this is not done automatically, to prevent
            surprises.
    """

    def dephase(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
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

    return transformer_primitives.map_operations(circuit, dephase)
