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
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

from cirq import protocols, value
from cirq.ops import raw_types

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
    """

    def __init__(
        self,
        sub_operation: 'cirq.Operation',
        conditions: Sequence[Union[str, 'cirq.MeasurementKey']],
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
        keys = tuple(value.MeasurementKey(c) if isinstance(c, str) else c for c in conditions)
        if isinstance(sub_operation, ClassicallyControlledOperation):
            keys += sub_operation._control_keys
            sub_operation = sub_operation._sub_operation
        self._control_keys: Tuple['cirq.MeasurementKey', ...] = keys
        self._sub_operation: 'cirq.Operation' = sub_operation

    def without_classical_controls(self) -> 'cirq.Operation':
        return self._sub_operation.without_classical_controls()

    @property
    def qubits(self):
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits):
        return self._sub_operation.with_qubits(*new_qubits).with_classical_controls(
            *self._control_keys
        )

    def _decompose_(self):
        result = protocols.decompose_once(self._sub_operation, NotImplemented)
        if result is NotImplemented:
            return NotImplemented

        return [ClassicallyControlledOperation(op, self._control_keys) for op in result]

    def _value_equality_values_(self):
        return (frozenset(self._control_keys), self._sub_operation)

    def __str__(self) -> str:
        keys = ', '.join(map(str, self._control_keys))
        return f'{self._sub_operation}.with_classical_controls({keys})'

    def __repr__(self):
        return (
            f'cirq.ClassicallyControlledOperation('
            f'{self._sub_operation!r}, {list(self._control_keys)!r})'
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._sub_operation)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._sub_operation)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ClassicallyControlledOperation':
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return new_sub_op.with_classical_controls(*self._control_keys)

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
            return NotImplemented  # coverage: ignore

        wire_symbols = sub_info.wire_symbols + ('^',) * len(self._control_keys)
        exponent_qubit_index = None
        if sub_info.exponent_qubit_index is not None:
            exponent_qubit_index = sub_info.exponent_qubit_index + len(self._control_keys)
        elif sub_info.exponent is not None:
            exponent_qubit_index = len(self._control_keys)
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols,
            exponent=sub_info.exponent,
            exponent_qubit_index=exponent_qubit_index,
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'conditions': self._control_keys,
            'sub_operation': self._sub_operation,
        }

    def _act_on_(self, args: 'cirq.ActOnArgs') -> bool:
        def not_zero(measurement):
            return any(i != 0 for i in measurement)

        measurements = [
            args.log_of_measurement_results.get(str(key), str(key)) for key in self._control_keys
        ]
        missing = [m for m in measurements if isinstance(m, str)]
        if missing:
            raise ValueError(f'Measurement keys {missing} missing when performing {self}')
        if all(not_zero(measurement) for measurement in measurements):
            protocols.act_on(self._sub_operation, args)
        return True

    def _with_measurement_key_mapping_(
        self, key_map: Dict[str, str]
    ) -> 'ClassicallyControlledOperation':
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return sub_operation.with_classical_controls(
            *[protocols.with_measurement_key_mapping(k, key_map) for k in self._control_keys]
        )

    def _with_key_path_prefix_(self, path: Tuple[str, ...]) -> 'ClassicallyControlledOperation':
        keys = [protocols.with_key_path_prefix(k, path) for k in self._control_keys]
        return self._sub_operation.with_classical_controls(*keys)

    def _with_rescoped_keys_(
        self,
        path: Tuple[str, ...],
        bindable_keys: FrozenSet['cirq.MeasurementKey'],
    ) -> 'ClassicallyControlledOperation':
        def map_key(key: value.MeasurementKey) -> value.MeasurementKey:
            for i in range(len(path) + 1):
                back_path = path[: len(path) - i]
                new_key = key.with_key_path_prefix(*back_path)
                if new_key in bindable_keys:
                    return new_key
            return key

        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return sub_operation.with_classical_controls(*[map_key(k) for k in self._control_keys])

    def _control_keys_(self) -> FrozenSet[value.MeasurementKey]:
        return frozenset(self._control_keys).union(protocols.control_keys(self._sub_operation))

    def _qasm_(self, args: 'cirq.QasmArgs') -> Optional[str]:
        args.validate_version('2.0')
        keys = [f'm_{key}!=0' for key in self._control_keys]
        all_keys = " && ".join(keys)
        return args.format('if ({0}) {1}', all_keys, protocols.qasm(self._sub_operation, args=args))
