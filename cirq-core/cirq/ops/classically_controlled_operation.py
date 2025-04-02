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
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import sympy

from cirq import protocols, value
from cirq.ops import op_tree, raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class ClassicallyControlledOperation(raw_types.Operation):
    """Augments existing operations to be conditionally executed.

    An operation that is classically controlled is executed iff all conditions
    evaluate to True. Currently the only condition type is a measurement key.
    A measurement key evaluates to True iff any qubit in the corresponding
    measurement operation evaluated to a non-zero value.

    This object is typically created via
     `operation.with_classical_controls(*conditions)`.

    Examples:

    >>> import cirq
    >>> a, b, c = cirq.LineQubit.range(3)
    >>> circuit1 = cirq.Circuit(
    ...     cirq.measure(a, key='control_key'),
    ...     cirq.X(b).with_classical_controls('control_key'))
    >>> print(circuit1)
    0: ─────────────M───────
                    ║
    1: ─────────────╫───X───
                    ║   ║
    control_key: ═══@═══^═══
    >>> circuit2 = cirq.Circuit([
    ...     cirq.measure(a, key='control_key1'),
    ...     cirq.measure(b, key='control_key2'),
    ...     cirq.X(c).with_classical_controls('control_key1', 'control_key2')])
    >>> print(circuit2)
                     ┌──┐
    0: ───────────────M─────────
                      ║
    1: ───────────────╫M────────
                      ║║
    2: ───────────────╫╫────X───
                      ║║    ║
    control_key1: ════@╬════^═══
                       ║    ║
    control_key2: ═════@════^═══
                     └──┘
    """

    def __init__(
        self,
        sub_operation: 'cirq.Operation',
        conditions: Sequence[Union[str, 'cirq.MeasurementKey', 'cirq.Condition', sympy.Basic]],
    ):
        """Initializes a `ClassicallyControlledOperation`.

        Multiple consecutive `ClassicallyControlledOperation` layers are
        squashed when possible, so one should not depend on a specific number
        of layers.

        Args:
            sub_operation: The operation to gate with a classical control
                condition.
            conditions: A sequence of measurement keys, or strings that can be
                parsed into measurement keys.

        Raises:
            ValueError: If an unsupported gate is being classically
                controlled.
        """
        if protocols.measurement_key_objs(sub_operation):
            raise ValueError(
                f'Cannot conditionally run operations with measurements: {sub_operation}'
            )
        conditions = tuple(conditions)
        if isinstance(sub_operation, ClassicallyControlledOperation):
            conditions += sub_operation._conditions
            sub_operation = sub_operation._sub_operation
        conds: List['cirq.Condition'] = []
        for c in conditions:
            if isinstance(c, str):
                c = value.MeasurementKey.parse_serialized(c)
            if isinstance(c, value.MeasurementKey):
                c = value.KeyCondition(c)
            if isinstance(c, sympy.Basic):
                c = value.SympyCondition(c)
            conds.append(c)
        self._conditions: Tuple['cirq.Condition', ...] = tuple(conds)
        self._sub_operation: 'cirq.Operation' = sub_operation

    @property
    def classical_controls(self) -> FrozenSet['cirq.Condition']:
        return frozenset(self._conditions).union(self._sub_operation.classical_controls)

    def without_classical_controls(self) -> 'cirq.Operation':
        return self._sub_operation.without_classical_controls()

    @property
    def qubits(self):
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits):
        return self._sub_operation.with_qubits(*new_qubits).with_classical_controls(
            *self._conditions
        )

    def _decompose_(self):
        return self._decompose_with_context_()

    def _decompose_with_context_(self, context: Optional['cirq.DecompositionContext'] = None):
        result = protocols.decompose_once(
            self._sub_operation, NotImplemented, flatten=False, context=context
        )
        if result is NotImplemented:
            return NotImplemented

        return op_tree.transform_op_tree(
            result, lambda op: ClassicallyControlledOperation(op, self._conditions)
        )

    def _value_equality_values_(self):
        return (frozenset(self._conditions), self._sub_operation)

    def __str__(self) -> str:
        keys = ', '.join(map(str, self._conditions))
        return f'{self._sub_operation}.with_classical_controls({keys})'

    def __repr__(self):
        return (
            f'cirq.ClassicallyControlledOperation('
            f'{self._sub_operation!r}, {list(self._conditions)!r})'
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._sub_operation)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._sub_operation)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ClassicallyControlledOperation':
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return ClassicallyControlledOperation(new_sub_op, self._conditions)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Optional['protocols.CircuitDiagramInfo']:
        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=args.known_qubit_count,
            known_qubits=args.known_qubits,
            use_unicode_characters=args.use_unicode_characters,
            precision=args.precision,
            label_map=args.label_map,
        )
        sub_info = protocols.circuit_diagram_info(self._sub_operation, sub_args, None)
        if sub_info is None:
            return NotImplemented  # pragma: no cover
        control_label_count = 0
        if args.label_map is not None:
            control_label_count = len({k for c in self._conditions for k in c.keys})
        wire_symbols = sub_info.wire_symbols + ('^',) * control_label_count
        if control_label_count == 0 or any(
            not isinstance(c, value.KeyCondition) for c in self._conditions
        ):
            wire_symbols = (
                wire_symbols[0]
                + '(conditions=['
                + ', '.join(str(c) for c in self._conditions)
                + '])',
            ) + wire_symbols[1:]
        exp_index = sub_info.exponent_qubit_index
        if exp_index is None:
            # None means at bottom, which means the last of the original wire symbols
            exp_index = len(sub_info.wire_symbols) - 1
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=sub_info.exponent, exponent_qubit_index=exp_index
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {'conditions': self._conditions, 'sub_operation': self._sub_operation}

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> bool:
        if all(c.resolve(sim_state.classical_data) for c in self._conditions):
            protocols.act_on(self._sub_operation, sim_state)
        return True

    def _with_measurement_key_mapping_(
        self, key_map: Mapping[str, str]
    ) -> 'ClassicallyControlledOperation':
        conditions = [protocols.with_measurement_key_mapping(c, key_map) for c in self._conditions]
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return sub_operation.with_classical_controls(*conditions)

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]) -> 'ClassicallyControlledOperation':
        conditions = [protocols.with_key_path_prefix(c, prefix) for c in self._conditions]
        sub_operation = protocols.with_key_path_prefix(self._sub_operation, prefix)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return sub_operation.with_classical_controls(*conditions)

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ) -> 'ClassicallyControlledOperation':
        conds = [protocols.with_rescoped_keys(c, path, bindable_keys) for c in self._conditions]
        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return sub_operation.with_classical_controls(*conds)

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        local_keys: FrozenSet['cirq.MeasurementKey'] = frozenset(
            k for condition in self._conditions for k in condition.keys
        )
        return local_keys.union(protocols.control_keys(self._sub_operation))

    def _qasm_(self, args: 'cirq.QasmArgs') -> Optional[str]:
        args.validate_version('2.0', '3.0')
        if len(self._conditions) > 1:
            raise ValueError('QASM does not support multiple conditions.')
        subop_qasm = protocols.qasm(self._sub_operation, args=args)
        if not self._conditions:
            return subop_qasm
        return f'if ({protocols.qasm(self._conditions[0], args=args)}) {subop_qasm}'
