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
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import sympy

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
        conditions: Sequence[Union[str, 'cirq.MeasurementKey', raw_types.Condition]],
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
        conds: List[raw_types.Condition] = []
        for c in conditions:
            if isinstance(c, str):
                c1 = parse_condition(c) or value.MeasurementKey.parse_serialized(c)
                if c1 is None:
                    raise ValueError(f"'{c}' is not a valid condition")
                c = c1
            if isinstance(c, value.MeasurementKey):
                c = raw_types.Condition(sympy.sympify('x0'), (c,))
            conds.append(c)
        self._conditions: Tuple[raw_types.Condition, ...] = tuple(conds)
        self._sub_operation: 'cirq.Operation' = sub_operation

    @property
    def classical_controls(self) -> FrozenSet[raw_types.Condition]:
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
        result = protocols.decompose_once(self._sub_operation, NotImplemented)
        if result is NotImplemented:
            return NotImplemented

        return [ClassicallyControlledOperation(op, self._conditions) for op in result]

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
        return new_sub_op.with_classical_controls(*self._conditions)

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

        wire_symbols = sub_info.wire_symbols + ('^',) * len(self._conditions)
        exponent_qubit_index = None
        if sub_info.exponent_qubit_index is not None:
            exponent_qubit_index = sub_info.exponent_qubit_index + len(self._conditions)
        elif sub_info.exponent is not None:
            exponent_qubit_index = len(self._conditions)
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols,
            exponent=sub_info.exponent,
            exponent_qubit_index=exponent_qubit_index,
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'conditions': self._conditions,
            'sub_operation': self._sub_operation,
        }

    def _act_on_(self, args: 'cirq.ActOnArgs') -> bool:
        for condition in self._conditions:
            keys, expr = condition.keys, condition.expr
            missing = [str(k) for k in keys if str(k) not in args.log_of_measurement_results]
            if missing:
                raise ValueError(f'Measurement keys {missing} missing when performing {self}')
            replacements = {
                f'x{i}': args.log_of_measurement_results[str(k)][0] for i, k in enumerate(keys)
            }
            result = expr.subs(replacements)
            if not result:
                return True
        protocols.act_on(self._sub_operation, args)
        return True

    def _with_measurement_key_mapping_(
        self, key_map: Dict[str, str]
    ) -> 'ClassicallyControlledOperation':
        def map_condition(condition: raw_types.Condition) -> raw_types.Condition:
            keys = [protocols.with_measurement_key_mapping(k, key_map) for k in condition.keys]
            return condition.with_keys(tuple(keys))

        conditions = [map_condition(c) for c in self._conditions]
        return self._sub_operation.with_classical_controls(*conditions)

    def _with_key_path_prefix_(self, path: Tuple[str, ...]) -> 'ClassicallyControlledOperation':
        def map_condition(condition: raw_types.Condition) -> raw_types.Condition:
            keys = tuple(protocols.with_key_path_prefix(k, path) for k in condition.keys)
            return condition.with_keys(keys)

        conditions = [map_condition(c) for c in self._conditions]
        return self._sub_operation.with_classical_controls(*conditions)

    def _control_keys_(self) -> FrozenSet[value.MeasurementKey]:
        local_keys: FrozenSet[value.MeasurementKey] = frozenset(
            k for condition in self._conditions for k in condition.keys
        )
        return local_keys.union(protocols.control_keys(self._sub_operation))

    def _qasm_(self, args: 'cirq.QasmArgs') -> Optional[str]:
        args.validate_version('2.0')
        keys = [f'm_{key}!=0' for key in self._conditions]
        all_keys = " && ".join(keys)
        return args.format('if ({0}) {1}', all_keys, protocols.qasm(self._sub_operation, args=args))


def parse_condition(s: str) -> Optional[raw_types.Condition]:
    in_key = False
    key_count = 0
    s_out = ''
    key_name = ''
    keys = []
    for c in s:
        if not in_key:
            if c == '{':
                in_key = True
            else:
                s_out += c
        else:
            if c == '}':
                symbol_name = f'x{key_count}'
                s_out += symbol_name
                keys.append(value.MeasurementKey.parse_serialized(key_name))
                key_name = ''
                key_count += 1
                in_key = False
            else:
                key_name += c
    expr = sympy.sympify(s_out)
    if len(expr.free_symbols) != len(keys):
        return None
    return raw_types.Condition(expr, tuple(keys))
