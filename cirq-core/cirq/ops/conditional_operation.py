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
    cast,
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
class ConditionalOperation(raw_types.Operation):
    """Augments existing operations to be conditionally executed."""

    def __init__(
        self,
        sub_operation: 'cirq.Operation',
        conditions: Sequence[Union[str, 'cirq.MeasurementKey']],
    ):
        keys = tuple(value.MeasurementKey(k) if isinstance(k, str) else k for k in conditions)
        self._control_keys = cast(Tuple[value.MeasurementKey, ...], keys)
        self._sub_operation = sub_operation

    def unconditionally(self) -> 'cirq.Operation':
        return self._sub_operation.unconditionally()

    @property
    def qubits(self):
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits):
        return self._sub_operation.with_qubits(*new_qubits).with_conditions(*self._control_keys)

    def _decompose_(self):
        result = protocols.decompose_once(self._sub_operation, NotImplemented)
        if result is NotImplemented:
            return NotImplemented

        return [ConditionalOperation(op, self._control_keys) for op in result]

    def _value_equality_values_(self):
        return (frozenset(self._control_keys), self._sub_operation)

    def __str__(self) -> str:
        keys = ', '.join(map(str, self._control_keys))
        return f'{self._sub_operation}.with_conditions({keys})'

    def __repr__(self):
        return f'cirq.ConditionalOperation({self._sub_operation!r}, {list(self._control_keys)!r})'

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._sub_operation)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._sub_operation)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ConditionalOperation':
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return new_sub_op.with_conditions(*self._control_keys)

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

        measurements = [args.log_of_measurement_results[str(key)] for key in self._control_keys]
        if all(not_zero(measurement) for measurement in measurements):
            protocols.act_on(self._sub_operation, args)
        return True

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'ConditionalOperation':
        keys = [protocols.with_measurement_key_mapping(k, key_map) for k in self._control_keys]
        return self._sub_operation.with_conditions(*keys)

    def _with_key_path_prefix_(self, path: Tuple[str, ...]) -> 'ConditionalOperation':
        keys = [protocols.with_key_path_prefix(k, path) for k in self._control_keys]
        return self._sub_operation.with_conditions(*keys)

    def _control_keys_(self) -> FrozenSet[value.MeasurementKey]:
        return frozenset(self._control_keys).union(protocols.control_keys(self._sub_operation))

    def _qasm_(self, args: 'cirq.QasmArgs') -> Optional[str]:
        args.validate_version('2.0')
        keys = [f'm_{key}!=0' for key in self._control_keys]
        all_keys = " && ".join(keys)
        return args.format('if ({0}) {1}', all_keys, protocols.qasm(self._sub_operation, args=args))
