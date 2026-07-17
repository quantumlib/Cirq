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
from typing import Any, Self, TYPE_CHECKING

import sympy

from cirq import protocols, value
from cirq.ops import classically_controlled_operation, raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class If(raw_types.Operation):
    """An operation that conditionally executes a sub-operation based on classical conditions.

    This operation decomposes into a `cirq.ClassicallyControlledOperation`.
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
        """Initializes the `If` operation.

        Args:
            condition: The condition(s) under which `sub_operation` should be applied.
                Can be a measurement key, string, condition object, sympy expression, or a
                sequence of these conditions.
            sub_operation: The operation (or tree of operations) to run when `condition` is satisfied.
            *more_operations: Additional operations to run when `condition` is satisfied. If provided,
                `sub_operation` and `more_operations` are combined into a `cirq.CircuitOperation`.

        Raises:
            ValueError: If `condition` sequence is empty, or if the sub-operation contains
                measurement keys.
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

        if not more_operations and isinstance(sub_operation, raw_types.Operation):
            if isinstance(sub_operation, If):
                self._conditions: tuple[cirq.Condition, ...] = (
                    conds_tuple + sub_operation.conditions
                )
                self._sub_operation: cirq.Operation = sub_operation.sub_operation
            elif isinstance(
                sub_operation, classically_controlled_operation.ClassicallyControlledOperation
            ):
                self._conditions = conds_tuple + sub_operation._conditions
                self._sub_operation = sub_operation._sub_operation
            else:
                self._conditions = conds_tuple
                self._sub_operation = sub_operation
        else:
            # Inline import to prevent circular dependency.
            from cirq.circuits import circuit, circuit_operation

            if more_operations:
                c = circuit.Circuit(sub_operation, *more_operations)
            else:
                c = circuit.Circuit(sub_operation)
            self._conditions = conds_tuple
            self._sub_operation = circuit_operation.CircuitOperation(c.freeze())

        if protocols.measurement_key_objs(self._sub_operation):
            raise ValueError(
                f'Cannot conditionally run operations with measurements: {self._sub_operation}'
            )

    @property
    def conditions(self) -> tuple[cirq.Condition, ...]:
        """All conditions that must be satisfied for the sub-operation to run."""
        return self._conditions

    @property
    def condition(self) -> cirq.Condition:
        """The condition that must be satisfied for the sub-operation to run.

        Raises:
            ValueError: If there are multiple conditions on this operation.
        """
        if len(self._conditions) != 1:
            raise ValueError(
                f'Operation has multiple conditions: {self._conditions}. '
                'Use `conditions` property instead.'
            )
        return self._conditions[0]

    @property
    def sub_operation(self) -> cirq.Operation:
        """The operation that is conditionally executed."""
        return self._sub_operation

    @property
    def classical_controls(self) -> frozenset[cirq.Condition]:
        return frozenset(self._conditions).union(self._sub_operation.classical_controls)

    def without_classical_controls(self) -> cirq.Operation:
        return self._sub_operation.without_classical_controls()

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._sub_operation.qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> Self:
        return If(self._conditions, self._sub_operation.with_qubits(*new_qubits))

    def _decompose_with_context_(
        self, *, context: cirq.DecompositionContext | None = None
    ) -> cirq.OP_TREE:
        return self._sub_operation.with_classical_controls(*self._conditions)

    def _decompose_(self) -> cirq.OP_TREE:
        return self._sub_operation.with_classical_controls(*self._conditions)

    def _value_equality_values_(self) -> Any:
        return self._conditions, self._sub_operation

    def __str__(self) -> str:
        if len(self._conditions) == 1:
            return f'If({self._conditions[0]}, {self._sub_operation})'
        keys = ', '.join(str(c) for c in self._conditions)
        return f'If(({keys}), {self._sub_operation})'

    def __repr__(self) -> str:
        if len(self._conditions) == 1:
            return f'cirq.If({self._conditions[0]!r}, {self._sub_operation!r})'
        return f'cirq.If({list(self._conditions)!r}, {self._sub_operation!r})'

    def _is_parameterized_(self) -> bool:
        return any(
            protocols.is_parameterized(c) for c in self._conditions
        ) or protocols.is_parameterized(self._sub_operation)

    def _parameter_names_(self) -> Set[str]:
        names: set[str] = set()
        for c in self._conditions:
            names.update(protocols.parameter_names(c))
        names.update(protocols.parameter_names(self._sub_operation))
        return names

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> If:
        new_conditions = [
            protocols.resolve_parameters(c, resolver, recursive) for c in self._conditions
        ]
        new_sub_op = protocols.resolve_parameters(self._sub_operation, resolver, recursive)
        return If(new_conditions, new_sub_op)

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> protocols.CircuitDiagramInfo | None:
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
            if len(self._conditions) == 1:
                cond_str = str(self._conditions[0])
            else:
                cond_str = ', '.join(str(c) for c in self._conditions)
            wire_symbols = (f'{wire_symbols[0]}(If={cond_str})', *wire_symbols[1:])
        exp_index = sub_info.exponent_qubit_index
        if exp_index is None:
            exp_index = len(sub_info.wire_symbols) - 1
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols, exponent=sub_info.exponent, exponent_qubit_index=exp_index
        )

    def _json_dict_(self) -> dict[str, Any]:
        return {'condition': list(self._conditions), 'sub_operation': self._sub_operation}

    def _act_on_(self, sim_state: cirq.SimulationStateBase) -> bool:
        if all(c.resolve(sim_state.classical_data) for c in self._conditions):
            protocols.act_on(self._sub_operation, sim_state)
        return True

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> If:
        conditions = [protocols.with_measurement_key_mapping(c, key_map) for c in self._conditions]
        sub_operation = protocols.with_measurement_key_mapping(self._sub_operation, key_map)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return If(conditions, sub_operation)

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> If:
        conditions = [protocols.with_key_path_prefix(c, prefix) for c in self._conditions]
        sub_operation = protocols.with_key_path_prefix(self._sub_operation, prefix)
        sub_operation = self._sub_operation if sub_operation is NotImplemented else sub_operation
        return If(conditions, sub_operation)

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> If:
        conds = [protocols.with_rescoped_keys(c, path, bindable_keys) for c in self._conditions]
        sub_operation = protocols.with_rescoped_keys(self._sub_operation, path, bindable_keys)
        return If(conds, sub_operation)

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
        if args.version == "2.0" and len(self._conditions) > 1:
            raise ValueError(
                'QASM 2.0 does not support multiple conditions. Consider exporting with QASM 3.0.'
            )
        subop_qasm = protocols.qasm(self._sub_operation, args=args, qubits=qubits, default=None)
        if subop_qasm is None:
            return None
        condition_qasm = " && ".join(protocols.qasm(c, args=args) for c in self._conditions)
        return f'if ({condition_qasm}) {subop_qasm}'
