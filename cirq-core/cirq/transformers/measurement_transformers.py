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

import itertools
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from cirq import ops, protocols, value
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements

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
        ValueError: If sympy-based classical conditions are used.
        NotImplementedError: When attempting to defer a measurement with a
            confusion map. (https://github.com/quantumlib/Cirq/issues/5482)
    """

    circuit = transformer_primitives.unroll_circuit_op(circuit, deep=True, tags_to_check=None)
    terminal_measurements = {op for _, op in find_terminal_measurements(circuit)}
    measurement_qubits: Dict['cirq.MeasurementKey', List['_MeasurementQid']] = {}

    def defer(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if op in terminal_measurements:
            return op
        gate = op.gate
        if isinstance(gate, ops.MeasurementGate):
            if gate.confusion_map:
                raise NotImplementedError(
                    "Deferring confused measurement is not implemented, but found "
                    f"measurement with key={gate.key} and non-empty confusion map."
                )
            key = value.MeasurementKey.parse_serialized(gate.key)
            targets = [_MeasurementQid(key, q) for q in op.qubits]
            measurement_qubits[key] = targets
            cxs = [ops.CX(q, target) for q, target in zip(op.qubits, targets)]
            xs = [ops.X(targets[i]) for i, b in enumerate(gate.full_invert_mask()) if b]
            return cxs + xs
        elif protocols.is_measurement(op):
            return [defer(op, None) for op in protocols.decompose_once(op)]
        elif op.classical_controls:
            new_op = op.without_classical_controls()
            for c in op.classical_controls:
                if isinstance(c, value.KeyCondition):
                    if c.key not in measurement_qubits:
                        raise ValueError(f'Deferred measurement for key={c.key} not found.')
                    qs = measurement_qubits[c.key]
                    if len(qs) == 1:
                        control_values: Any = range(1, qs[0].dimension)
                    else:
                        all_values = itertools.product(*[range(q.dimension) for q in qs])
                        anything_but_all_zeros = tuple(itertools.islice(all_values, 1, None))
                        control_values = ops.SumOfProducts(anything_but_all_zeros)
                    new_op = new_op.controlled_by(*qs, control_values=control_values)
                else:
                    raise ValueError('Only KeyConditions are allowed.')
            return new_op
        return op

    circuit = transformer_primitives.map_operations_and_unroll(
        circuit=circuit,
        map_func=defer,
        tags_to_ignore=context.tags_to_ignore if context else (),
        raise_if_add_qubits=False,
    ).unfreeze()
    for k, qubits in measurement_qubits.items():
        circuit.append(ops.measure(*qubits, key=k))
    return circuit


@transformer_api.transformer
def dephase_measurements(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = transformer_api.TransformerContext(deep=True),
) -> 'cirq.Circuit':
    """Changes all measurements to a dephase operation.

    This transformer is useful when using a density matrix simulator, when
    wishing to calculate the final density matrix of a circuit and not simulate
    the measurements themselves.

    Args:
        circuit: The circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options
            for transformers. The default has `deep=True` to ensure
            measurements at all levels are dephased.
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
        return op

    ignored = () if context is None else context.tags_to_ignore
    return transformer_primitives.map_operations(
        circuit, dephase, deep=context.deep if context else True, tags_to_ignore=ignored
    ).unfreeze()


@transformer_api.transformer
def drop_terminal_measurements(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = transformer_api.TransformerContext(deep=True),
) -> 'cirq.Circuit':
    """Removes terminal measurements from a circuit.

    This transformer is helpful when trying to capture the final state vector
    of a circuit with many terminal measurements, as simulating the circuit
    with those measurements in place would otherwise collapse the final state.

    Args:
        circuit: The circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options
            for transformers. The default has `deep=True`, as "terminal
            measurements" is ill-defined without inspecting subcircuits;
            passing a context with `deep=False` will return an error.
    Returns:
        A copy of the circuit, with identity or X gates in place of terminal
        measurements.
    Raises:
        ValueError: if the circuit contains non-terminal measurements, or if
            the provided context has`deep=False`.
    """

    if context is None or not context.deep:
        raise ValueError(
            'Context has `deep=False`, but `deep=True` is required to drop terminal measurements.'
        )

    if not circuit.are_all_measurements_terminal():
        raise ValueError('Circuit contains a non-terminal measurement.')

    def flip_inversion(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if isinstance(op.gate, ops.MeasurementGate):
            return [
                ops.X(q) if b else ops.I(q) for q, b in zip(op.qubits, op.gate.full_invert_mask())
            ]
        return op

    ignored = () if context is None else context.tags_to_ignore
    return transformer_primitives.map_operations(
        circuit, flip_inversion, deep=context.deep if context else True, tags_to_ignore=ignored
    ).unfreeze()
