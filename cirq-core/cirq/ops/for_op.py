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
from collections.abc import Iterable, Mapping, Sequence
from types import NotImplementedType
from typing import Any, cast, TYPE_CHECKING

import numpy as np
import sympy

from cirq import _compat, protocols, study, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class For(raw_types.Operation):
    """An operation that iterates over values, executing a sub-operation for each value.

    Note: This is an experimental function designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change between versions.
    """

    def __init__(
        self,
        loop_var_or_values: (
            str
            | sympy.Symbol
            | int
            | Iterable[Any]
            | sympy.Basic
            | tuple[str | sympy.Symbol | None, int | Iterable[Any] | sympy.Basic]
            | None
        ) = None,
        values_or_sub_operation: (
            int | Iterable[Any] | sympy.Basic | cirq.Operation | cirq.OP_TREE | None
        ) = None,
        sub_operation: cirq.Operation | cirq.OP_TREE | None = None,
        *more_operations: cirq.Operation | cirq.OP_TREE,
        loop_var: str | sympy.Symbol | None = None,
        values: int | Iterable[Any] | sympy.Basic | None = None,
    ):
        """Initializes the `For` operation.

        Args:
            loop_var_or_values: Either the loop variable (as a string or SymPy
                Symbol), an integer/iterable of values to iterate over, or a
                tuple `(loop_var, values)`.
            values_or_sub_operation: Either the values to iterate over (if
                `loop_var_or_values` was a variable), or the sub-operation to
                execute (if `loop_var_or_values` was the values).
            sub_operation: The operation (or tree of operations) to run on each
                iteration when `loop_var` and `values` are passed as the first
                two arguments.
            *more_operations: Additional operations to run in each iteration.
                If provided, `sub_operation` and `more_operations` are combined
                into a `cirq.CircuitOperation`.
            loop_var: Optional explicit loop variable keyword argument.
            values: Optional explicit values keyword argument.

        Raises:
            ValueError: If values or sub-operation are missing, or if iteration
                count is negative.
            TypeError: If an unrecognized type is provided for `loop_var` or
                `values`.
        """
        from cirq.circuits import Circuit, CircuitOperation

        def is_op_or_ops(obj: Any) -> bool:
            if isinstance(obj, (raw_types.Operation, Circuit)):
                return True
            if isinstance(obj, (list, tuple)) and obj:
                return all(isinstance(x, (raw_types.Operation, Circuit)) for x in obj)
            return False

        if loop_var is not None or values is not None:
            parsed_loop_var = loop_var
            parsed_values = values
            parsed_sub_op = (
                sub_operation
                if sub_operation is not None
                else (
                    values_or_sub_operation
                    if values_or_sub_operation is not None
                    else loop_var_or_values
                )
            )
            parsed_more_ops = more_operations
        elif is_op_or_ops(values_or_sub_operation):
            if (
                isinstance(loop_var_or_values, tuple)
                and len(loop_var_or_values) == 2
                and not is_op_or_ops(loop_var_or_values[0])
            ):
                parsed_loop_var, parsed_values = loop_var_or_values
            else:
                parsed_loop_var = None
                parsed_values = loop_var_or_values
            parsed_sub_op = values_or_sub_operation
            parsed_more_ops = (
                (sub_operation, *more_operations) if sub_operation is not None else more_operations
            )
        else:
            parsed_loop_var = loop_var_or_values
            parsed_values = values_or_sub_operation
            parsed_sub_op = sub_operation
            parsed_more_ops = more_operations

        # Normalize loop_var
        if parsed_loop_var is None:
            self._loop_var: sympy.Symbol | None = None
        elif isinstance(parsed_loop_var, str):
            self._loop_var = sympy.Symbol(parsed_loop_var)
        elif isinstance(parsed_loop_var, sympy.Symbol):
            self._loop_var = parsed_loop_var
        else:
            raise TypeError(f"Unrecognized loop_var type: {type(parsed_loop_var)}")

        # Normalize values
        if parsed_values is None:
            raise ValueError("Iteration values must be provided.")
        elif isinstance(parsed_values, (int, numbers.Integral, np.integer)):
            int_val = int(parsed_values)
            if int_val < 0:
                raise ValueError(f"Iteration count must be non-negative: {int_val}")
            self._values: tuple[Any, ...] | sympy.Basic = tuple(range(int_val))
        elif isinstance(parsed_values, sympy.Basic):
            self._values = parsed_values
        elif isinstance(parsed_values, Iterable):
            self._values = tuple(parsed_values)
        else:
            raise TypeError(f"Unrecognized values type: {type(parsed_values)}")

        # Normalize sub_operation
        if parsed_sub_op is None:
            raise ValueError("At least one sub-operation must be provided.")

        if parsed_more_ops or not isinstance(parsed_sub_op, raw_types.Operation):
            from cirq.circuits import Circuit, CircuitOperation

            c = Circuit(cast('cirq.OP_TREE', parsed_sub_op), *parsed_more_ops)
            self._sub_operation: cirq.Operation = CircuitOperation(c.freeze())
        else:
            self._sub_operation = parsed_sub_op

    @property
    def loop_var(self) -> sympy.Symbol | None:
        """The loop variable, or None if no variable was specified."""
        return self._loop_var

    @property
    def values(self) -> tuple[Any, ...] | sympy.Basic:
        """The sequence of values or symbolic range to iterate over."""
        return self._values

    @property
    def sub_operation(self) -> cirq.Operation:
        """The operation executed in each iteration."""
        return self._sub_operation

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> For:
        return For(self._loop_var, self._values, self._sub_operation.with_qubits(*new_qubits))

    @property
    def classical_controls(self) -> frozenset[cirq.Condition]:
        return self._sub_operation.classical_controls

    def without_classical_controls(self) -> For:
        return For(self._loop_var, self._values, self._sub_operation.without_classical_controls())

    def _decompose_with_context_(
        self, *, context: cirq.DecompositionContext | None = None
    ) -> cirq.OP_TREE:
        return self._decompose_()

    def _decompose_(self) -> cirq.OP_TREE:
        if isinstance(self._values, sympy.Basic) and protocols.is_parameterized(self._values):
            return NotImplemented
        ops = []
        for val in self._values:
            if self._loop_var is not None:
                step_op = protocols.resolve_parameters(self._sub_operation, {self._loop_var: val})
            else:
                step_op = self._sub_operation
            ops.append(step_op)
        return ops

    def _value_equality_values_(self) -> Any:
        return self._loop_var, self._values, self._sub_operation

    def __str__(self) -> str:
        val_repr = list(self._values) if isinstance(self._values, tuple) else self._values
        if self._loop_var is not None:
            return f'For({self._loop_var}, {val_repr}, {self._sub_operation})'
        return f'For({val_repr}, {self._sub_operation})'

    def __repr__(self) -> str:
        val_repr = list(self._values) if isinstance(self._values, tuple) else self._values
        if self._loop_var is not None:
            return (
                f'cirq.For({_compat.proper_repr(self._loop_var)}, '
                f'{val_repr!r}, {self._sub_operation!r})'
            )
        return f'cirq.For({val_repr!r}, {self._sub_operation!r})'

    @_compat.cached_method
    def _parameter_names_(self) -> frozenset[str]:
        sub_params = set(protocols.parameter_names(self._sub_operation))
        if self._loop_var is not None:
            sub_params.discard(self._loop_var.name)
        val_params = protocols.parameter_names(self._values)
        return frozenset(sub_params | val_params)

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return bool(self._parameter_names_())

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> For:
        if self._loop_var is not None and self._loop_var.name in resolver.param_dict:
            sub_resolver = study.ParamResolver(
                {k: v for k, v in resolver.param_dict.items() if k != self._loop_var.name}
            )
            new_sub_op = protocols.resolve_parameters(self._sub_operation, sub_resolver, recursive)
        else:
            new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)

        if isinstance(self._values, sympy.Basic):
            new_values = protocols.resolve_parameters(self._values, resolver, recursive)
            if not protocols.is_parameterized(new_values):
                try:
                    int_val = int(new_values)
                    if int_val < 0:
                        raise ValueError(f"Iteration count must be non-negative: {int_val}")
                    new_values = tuple(range(int_val))
                except (TypeError, ValueError):
                    pass
        elif isinstance(self._values, (list, tuple)):
            new_values = tuple(
                (
                    protocols.resolve_parameters(v, resolver, recursive)
                    if (isinstance(v, sympy.Basic) or protocols.is_parameterized(v))
                    else v
                )
                for v in self._values
            )
        else:  # pragma: no cover
            new_values = protocols.resolve_parameters(self._values, resolver, recursive)

        return For(self._loop_var, new_values, new_sub_op)

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
        if sub_info is None:
            return NotImplemented  # pragma: no cover
        val_str = str(list(self._values)) if isinstance(self._values, tuple) else str(self._values)
        for_label = f'{self._loop_var} in {val_str}' if self._loop_var is not None else val_str
        wire_symbols = (f'{sub_info.wire_symbols[0]}(For={for_label})', *sub_info.wire_symbols[1:])
        exp_index = sub_info.exponent_qubit_index
        if exp_index is None:
            exp_index = len(sub_info.wire_symbols) - 1
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=sub_info.exponent, exponent_qubit_index=exp_index
        )

    def _json_dict_(self) -> dict[str, Any]:
        val_repr = list(self._values) if isinstance(self._values, tuple) else self._values
        return {
            'loop_var': self._loop_var,
            'values': val_repr,
            'sub_operation': self._sub_operation,
        }

    def _act_on_(self, sim_state: cirq.SimulationStateBase) -> bool:
        if isinstance(self._values, sympy.Basic) and protocols.is_parameterized(self._values):
            raise ValueError(
                f"Cannot simulate parameterized For loop with symbolic bounds: {self._values}"
            )
        for val in self._values:
            if self._loop_var is not None:
                step_op = protocols.resolve_parameters(self._sub_operation, {self._loop_var: val})
            else:
                step_op = self._sub_operation
            protocols.act_on(step_op, sim_state)
        return True

    @_compat.cached_method
    def _measurement_key_names_(self) -> frozenset[str]:
        return protocols.measurement_key_names(self._sub_operation)

    @_compat.cached_method
    def _measurement_key_objs_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.measurement_key_objs(self._sub_operation)

    @_compat.cached_method
    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self._sub_operation)

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> For:
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return For(self._loop_var, self._values, sub_operation)

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> For:
        sub_operation = protocols.with_key_path_prefix(self._sub_operation, prefix)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return For(self._loop_var, self._values, sub_operation)

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> For:
        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return For(self._loop_var, self._values, sub_operation)

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.control_keys(self._sub_operation)

    def _qasm_(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        if args is None:
            from cirq.protocols.qasm import QasmArgs

            args = QasmArgs()
        if args.version == '2.0':
            raise ValueError(
                'QASM 2.0 does not support for loops. Consider exporting with QASM 3.0.'
            )
        args.validate_version('3.0')
        subop_qasm = protocols.qasm(self._sub_operation, args=args, qubits=qubits, default=None)
        if subop_qasm is None:
            return None
        var_name = self._loop_var.name if self._loop_var is not None else '_'
        if isinstance(self._values, (list, tuple)):
            val_items = ', '.join(str(v) for v in self._values)
            return f'for int {var_name} in {{{val_items}}} {subop_qasm}'
        return f'for int {var_name} in [{self._values}] {subop_qasm}'
