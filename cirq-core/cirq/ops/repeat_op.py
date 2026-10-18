# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from types import NotImplementedType
from typing import Any, TYPE_CHECKING

from cirq import _compat, protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class Repeat(raw_types.Operation):
    """An operation that executes a sub-operation a fixed number of times.

    This operation decomposes into a `cirq.CircuitOperation` with `repetitions=count`.

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0.  The interface for this class is subject to change between versions.
    """

    def __init__(
        self,
        count: int,
        sub_operation: cirq.Operation | cirq.OP_TREE,
        *more_operations: cirq.Operation | cirq.OP_TREE,
    ):
        """Initializes the `Repeat` operation.

        Args:
            count: The number of times `sub_operation` is executed. May be zero.
            sub_operation: The operation (or tree of operations) to repeat.
            *more_operations: Additional operations to repeat. If provided,
                `sub_operation` and `more_operations` are combined into a
                `cirq.CircuitOperation`.

        Raises:
            TypeError: If `count` is not an integer.
            ValueError: If `count` is negative.
        """
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError(f'count must be an integer, got: {count!r}')
        if count < 0:
            raise ValueError(f'count must be non-negative, got: {count!r}')
        self._count: int = int(count)

        if more_operations or not isinstance(sub_operation, raw_types.Operation):
            # Multiple operations: wrap in a CircuitOperation

            # Inline import to prevent circular dependency.
            from cirq.circuits import Circuit, CircuitOperation

            c = Circuit(sub_operation, *more_operations)
            self._sub_operation: cirq.Operation = CircuitOperation(c.freeze())
        else:
            # Single operation: preserve sub_operation as-is (including nested
            # Repeat, While, If, or ClassicallyControlledOperation).
            self._sub_operation = sub_operation

    @property
    def count(self) -> int:
        """The number of times the sub-operation is executed."""
        return self._count

    @property
    def sub_operation(self) -> cirq.Operation:
        """The operation that is repeated."""
        return self._sub_operation

    @property
    def classical_controls(self) -> frozenset[cirq.Condition]:
        return self._sub_operation.classical_controls

    def without_classical_controls(self) -> cirq.Operation:
        return Repeat(self._count, self._sub_operation.without_classical_controls())

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> Repeat:
        return Repeat(self._count, self._sub_operation.with_qubits(*new_qubits))

    def _decompose_with_context_(
        self, *, context: cirq.DecompositionContext | None = None
    ) -> cirq.OP_TREE:
        return self._decompose_()

    def _decompose_(self) -> cirq.OP_TREE:
        # Inline import to prevent circular dependency.
        from cirq.circuits import CircuitOperation, FrozenCircuit

        # use_repetition_ids is passed explicitly so that behavior does not depend on the
        # CircuitOperation default; measurement keys are shared across iterations.
        return CircuitOperation(
            FrozenCircuit(self._sub_operation), repetitions=self._count, use_repetition_ids=False
        )

    def _value_equality_values_(self) -> Any:
        return self._count, self._sub_operation

    def __str__(self) -> str:
        prefix = f'Repeat({self._count}, '
        sub_op_str = str(self._sub_operation).replace('\n', '\n' + ' ' * len(prefix))
        return f'{prefix}{sub_op_str})'

    def __repr__(self) -> str:
        return f'cirq.Repeat({self._count!r}, {self._sub_operation!r})'

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._sub_operation)

    @_compat.cached_method
    def _parameter_names_(self) -> frozenset[str]:
        return frozenset(protocols.parameter_names(self._sub_operation))

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> Repeat:
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return Repeat(self._count, new_sub_op)

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> protocols.CircuitDiagramInfo | NotImplementedType:
        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=args.known_qubit_count,
            known_qubits=args.known_qubits,
            use_unicode_characters=args.use_unicode_characters,
            precision=args.precision,
            label_map=args.label_map,
        )
        sub_info = protocols.circuit_diagram_info(self._sub_operation, sub_args, None)
        if sub_info is None or not sub_info.wire_symbols:
            return NotImplemented
        wire_symbols = (
            f'{sub_info.wire_symbols[0]}(Repeat={self._count})',
            *sub_info.wire_symbols[1:],
        )
        exp_index = sub_info.exponent_qubit_index
        if exp_index is None:
            exp_index = len(sub_info.wire_symbols) - 1
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=sub_info.exponent, exponent_qubit_index=exp_index
        )

    def _json_dict_(self) -> dict[str, Any]:
        return {'count': self._count, 'sub_operation': self._sub_operation}

    @_compat.cached_method
    def _measurement_key_names_(self) -> frozenset[str]:
        return protocols.measurement_key_names(self._sub_operation)

    @_compat.cached_method
    def _measurement_key_objs_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.measurement_key_objs(self._sub_operation)

    @_compat.cached_method
    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self._sub_operation)

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> Repeat:
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return Repeat(self._count, sub_operation)

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> Repeat:
        sub_operation = protocols.with_key_path_prefix(self._sub_operation, prefix)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return Repeat(self._count, sub_operation)

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> Repeat:
        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return Repeat(self._count, sub_operation)

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.control_keys(self._sub_operation)

    def _qasm_(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        if args is None:
            args = protocols.QasmArgs()
        args.validate_version('2.0', '3.0')
        if args.version == "2.0":
            raise ValueError('QASM 2.0 does not support loops. Consider exporting with QASM 3.0.')
        subop_qasm = protocols.qasm(self._sub_operation, args=args, qubits=qubits, default=None)
        if subop_qasm is None:
            return None
        if self._count == 0:
            return ''
        body = ''.join(f'  {line}\n' for line in subop_qasm.splitlines())
        return f'for int _i in [0:{self._count - 1}] {{\n{body}}}\n'
