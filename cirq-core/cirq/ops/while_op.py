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

from collections.abc import Mapping, Sequence, Set
from types import NotImplementedType
from typing import Any, TYPE_CHECKING

import sympy

from cirq import _compat, protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class While(raw_types.Operation):
    """An operation that repeatedly executes a sub-operation while a classical condition is True.

    In contrast to If, this operation does NOT decompose to a `cirq.ClassicallyControlledOperation`.

    Note: This is an experimental function designed as part of a prototype
    for Cirq 2.0.  The interface for this class is subject to change between versions.
    """

    def __init__(
        self,
        condition: (
            str
            | cirq.MeasurementKey
            | cirq.Condition
            | sympy.Basic
            | Sequence[str | cirq.MeasurementKey | cirq.Condition | sympy.Basic]
        ),
        sub_operation: cirq.Operation | cirq.OP_TREE,
        *more_operations: cirq.Operation | cirq.OP_TREE,
    ):
        """Initializes the `While` operation.

        Args:
            condition: The condition(s) under which `sub_operation` should be
                applied.  Can be a measurement key, string, condition object,
                sympy expression, or a sequence of these conditions. `sub_operation`
                will be continuously applied until `condition` is no longer true.
            sub_operation: The operation (or tree of operations) to run when
                `condition` is satisfied.
            *more_operations: Additional operations to run when `condition`
                is satisfied. If provided, `sub_operation` and `more_operations`
                are combined into a `cirq.CircuitOperation`.

        Raises:
            ValueError: If `condition` sequence is empty,
                or if the sub-operation contains measurement keys.
            TypeError: If an unrecognized condition type is provided.
        """
        if isinstance(condition, (str, value.MeasurementKey, value.Condition, sympy.Basic)):
            raw_conditions: Sequence[Any] = (condition,)
        elif isinstance(condition, Sequence):
            raw_conditions = condition
        else:
            raise TypeError(f"Unrecognized condition type: {type(condition)}")

        conds: list[cirq.Condition] = []
        for c in raw_conditions:
            if isinstance(c, str):
                c = value.MeasurementKey.parse_serialized(c)
            if isinstance(c, value.MeasurementKey):
                c = value.KeyCondition(c)
            if isinstance(c, sympy.Basic):
                c = value.SympyCondition(c)
            if not isinstance(c, value.Condition):
                raise TypeError(f"Unrecognized condition type: {type(c)}")
            conds.append(c)

        if not conds:
            raise ValueError("At least one condition must be provided.")
        conds_tuple = tuple(conds)

        if more_operations or not isinstance(sub_operation, raw_types.Operation):
            # Multiple operations: wrap in a CircuitOperation

            # Inline import to prevent circular dependency.
            from cirq.circuits import Circuit, CircuitOperation

            c = Circuit(sub_operation, *more_operations)
            self._conditions: tuple[cirq.Condition, ...] = conds_tuple
            self._sub_operation: cirq.Operation = CircuitOperation(c.freeze())
        else:
            # Single operation: preserve sub_operation as-is (including nested
            # While, If, or ClassicallyControlledOperation) so the outer While's
            # termination condition is not altered.
            self._conditions = conds_tuple
            self._sub_operation = sub_operation

        # In contrast with "If", measurements must be allowed, otherwise the While
        # loop will get stuck in an infinite loop b/c the break condition will never
        # change.

    @property
    def conditions(self) -> tuple[cirq.Condition, ...]:
        """All conditions that must be satisfied for the sub-operation to run."""
        return self._conditions

    @property
    def sub_operation(self) -> cirq.Operation:
        """The operation that is conditionally executed."""
        return self._sub_operation

    @property
    def classical_controls(self) -> frozenset[cirq.Condition]:
        return frozenset(self._conditions).union(self._sub_operation.classical_controls)

    def without_classical_controls(self) -> cirq.Operation:
        raise ValueError('Cannot remove classical controls from a While operation.')

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> While:
        return While(self._conditions, self._sub_operation.with_qubits(*new_qubits))

    # Note: We intentionaly omit _decompose_with_context_ and _decompose here (as
    # opposed to If, which requires it) because this While loop cannot be
    # statically decomposed ahead of time. Instead, we have to tell cirq that this
    # construct isn't unitary.
    def _has_unitary_(self) -> bool:
        return False

    def _value_equality_values_(self) -> Any:
        return self._conditions, self._sub_operation

    def __str__(self) -> str:
        if len(self._conditions) == 1:
            return f'While({self._conditions[0]}, {self._sub_operation})'
        keys = ', '.join(str(c) for c in self._conditions)
        return f'While([{keys}], {self._sub_operation})'

    def __repr__(self) -> str:
        if len(self._conditions) == 1:
            return f'cirq.While({self._conditions[0]!r}, {self._sub_operation!r})'
        return f'cirq.While({list(self._conditions)!r}, {self._sub_operation!r})'

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._conditions) or protocols.is_parameterized(
            self._sub_operation
        )

    @_compat.cached_method
    def _parameter_names_(self) -> Set[str]:
        return frozenset(protocols.parameter_names(self._sub_operation)).union(
            *(protocols.parameter_names(c) for c in self._conditions)
        )

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> While:
        new_conditions = [
            protocols.resolve_parameters(c, resolver, recursive) for c in self._conditions
        ]
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return While(new_conditions, new_sub_op)

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
        control_label_count = 0
        if args.label_map is not None:
            # If self._sub_operation already measures or is controlled by a key
            # that is also used in self._conditions (e.g.,
            # cirq.While('a', cirq.measure(q0, key='a'))), sub_info.wire_symbols
            # already includes '@' or '^' for that key and Cirq's diagram drawer
            # deduplicates classical wire rows between measurement_key_objs(op)
            # and control_keys(op). Subtract sub_keys so we only append '^' for
            # control keys that don't already have a symbol from sub_info.
            sub_keys = protocols.measurement_key_objs(self._sub_operation).union(
                protocols.control_keys(self._sub_operation)
            )
            control_label_count = len({k for c in self._conditions for k in c.keys} - sub_keys)
        wire_symbols = sub_info.wire_symbols + ('^',) * control_label_count
        if len(self._conditions) == 1:
            cond_str = str(self._conditions[0])
        else:
            cond_str = ', '.join(str(c) for c in self._conditions)
        wire_symbols = (f'{wire_symbols[0]}(While={cond_str})', *wire_symbols[1:])
        exp_index = sub_info.exponent_qubit_index
        if exp_index is None:
            exp_index = len(sub_info.wire_symbols) - 1
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=sub_info.exponent, exponent_qubit_index=exp_index
        )

    def _json_dict_(self) -> dict[str, Any]:
        return {'condition': list(self._conditions), 'sub_operation': self._sub_operation}

    def _act_on_(self, sim_state: cirq.SimulationStateBase) -> bool:
        while all(c.resolve(sim_state.classical_data) for c in self._conditions):
            protocols.act_on(self._sub_operation, sim_state)
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

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> While:
        conditions = [protocols.with_measurement_key_mapping(c, key_map) for c in self._conditions]
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return While(conditions, sub_operation)

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> While:
        conditions = [protocols.with_key_path_prefix(c, prefix) for c in self._conditions]
        sub_operation = protocols.with_key_path_prefix(self._sub_operation, prefix)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return While(conditions, sub_operation)

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> While:
        conds = [protocols.with_rescoped_keys(c, path, bindable_keys) for c in self._conditions]
        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return While(conds, sub_operation)

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        local_keys: frozenset[cirq.MeasurementKey] = frozenset(
            k for condition in self._conditions for k in condition.keys
        )
        return local_keys.union(protocols.control_keys(self._sub_operation))

    def _qasm_(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        if args is None:
            from cirq.protocols.qasm import QasmArgs

            args = QasmArgs()
        args.validate_version('2.0', '3.0')
        if args.version == "2.0":
            raise ValueError(
                'QASM 2.0 does not support while loops. Consider exporting with QASM 3.0.'
            )
        subop_qasm = protocols.qasm(self._sub_operation, args=args, qubits=qubits, default=None)
        if subop_qasm is None:
            return None
        condition_qasm = " && ".join(protocols.qasm(c, args=args) for c in self._conditions)
        return f'while ({condition_qasm}) {subop_qasm}'
