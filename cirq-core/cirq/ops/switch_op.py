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

from collections.abc import Iterable, Sequence
from types import NotImplementedType
from typing import Any, TYPE_CHECKING

from cirq import _compat, protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


def _op_or_circuit_op(val: raw_types.Operation | cirq.OP_TREE) -> raw_types.Operation:
    from cirq.circuits import CircuitOperation, FrozenCircuit

    if isinstance(val, raw_types.Operation):
        return val
    val_circuit = FrozenCircuit(val)
    return CircuitOperation(val_circuit)


@value.value_equality
class Case:
    """One branch of a `cirq.Switch`.

    A `Case` is a pair of values `values` and a sequence of operations `then`. When used with
    `Switch`, it represents one branch of operations to run when any of its values match the
    `MeasurementKey` associated with `Switch`.
    """

    values: tuple[value.TParamVal, ...]
    then: raw_types.Operation | cirq.OP_TREE

    def __init__(
        self, values: tuple[value.TParamVal, ...], then: raw_types.Operation | cirq.OP_TREE
    ):
        if len(values) == 0:
            raise ValueError('Case object needs at least one condition.')
        self.values = values
        self.then = _op_or_circuit_op(then)

    def get_condition(self, k: cirq.MeasurementKey) -> cirq.OrCondition:
        """Returns the compiled OrCondition object given a key.

        Args:
            k: The key to match against.
        """
        return value.OrCondition(
            tuple(
                value.BitMaskKeyCondition(k, target_value=val, equal_target=True)
                for val in self.values
            )
        )

    def check(self, k: cirq.MeasurementKey, classical_data: cirq.ClassicalDataStoreReader) -> bool:
        """Checks to see if this case matches the evaluated classical data.

        Args:
            k: The key to match against.
            classical_data: The classical data to evaluate against.
        """
        resolved_value = self.get_condition(k).resolve(classical_data)
        return resolved_value

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.then)

    @_compat.cached_method
    def _parameter_names_(self) -> frozenset[str]:
        return protocols.parameter_names(self.values) | protocols.parameter_names(self.then)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> Switch:
        return Case(
            tuple(protocols.resolve_parameters(val, resolver, recursive) for val in self.values),
            protocols.resolve_parameters(self.then, resolver, recursive),
        )

    def __repr__(self) -> str:
        return f'cirq.Case({self.values!r}, {self.then!r})'

    def _qasm_(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        if args is None:
            from cirq.protocols.qasm import QasmArgs

            args = QasmArgs()

        args.validate_version('3.1')
        return self.qasm(args=args, qubits=qubits)

    def qasm(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        int_const_list_expr = ', '.join(str(val) for val in self.values)

        ops_qasm = protocols.qasm(self.then, args=args, qubits=qubits)
        if ops_qasm is None:
            return None  # pragma: no cover
        ops_qasm = ''.join(f'\n    {line}' for line in ops_qasm.strip().split('\n'))
        return f'case {int_const_list_expr} {{{ops_qasm}\n}}'

    def _value_equality_values_(self) -> Any:
        return self.values, self.then

    def _json_dict_(self) -> dict[str, Any]:
        return {'values': self.values, 'then': self.then}

    @classmethod
    def _from_json_dict_(cls, values, then, **kwargs) -> Case:
        return cls(tuple(values), then)


@value.value_equality
class Switch(raw_types.Operation):
    """A multi-branch operation for Cirq.

    `Switch` is an operation that takes in a `control` clasiscal value, a series of `Case` objects,
    and an optional `default` operation seqeunce. It allows for multiple branching when running a
    quantum circuit.

    Switch behaves in the following way:
        * It takes in a `control` MeasurementKey.
        * It sequentially iterates over all `Case` objects in insertion order, and checks whether
          its values matches the recorded value at `control`.
            * If any of the values match `control`, the operation sequence associated with the
              `Case` object is run.
            * Multiple `Case` objects may contain similar values. Multiple `Case` objects may match
              `control`, at which point each of them will be run.
        * If a `default` operation sequence is supplied, then it is run if none of the `Case`
          objects match.
    """

    def __init__(
        self,
        control: str | cirq.MeasurementKey,
        cases: Iterable[Case],
        default: raw_types.Operation | cirq.OP_TREE | None = None,
    ):
        """Initializes the `Switch` operation.

        Args:
            control: What to switch on. A measurement key (or its string form),
                whose big-endian integer value is used, or a SymPy expression over
                measurement keys, e.g. `sympy.Symbol('a') + 2 * sympy.Symbol('b')`.
                `sympy.IndexedBase` may be used to address individual bits.
            cases: The branches, in priority order. Either a mapping from case value
                to operation(s), or an iterable of `cirq.Case`s / `(value, op)` pairs.
                Case values are integers or SymPy parameters; parameters must be
                resolved before the operation is simulated or decomposed.
            default: Operation(s) to run when no case matches. If omitted, nothing
                runs in that situation.

        Raises:
            ValueError: If no cases are given, two cases share the same integer
                value, a parameter name collides with a control measurement key,
                or a branch contains measurements.
            TypeError: If `control` or a case value has an unsupported type.
        """
        self._control: value.MeasurementKey = (
            control if isinstance(control, value.MeasurementKey) else value.MeasurementKey(control)
        )
        self._cases: tuple[Case, ...] = tuple(cases)
        if default is not None:
            default = _op_or_circuit_op(default)
        self._default: raw_types.Operation | None = default

        if not self._cases:
            raise ValueError('At least one case must be provided.')

        qubits: set[cirq.Qid] = {q for c in self._cases for q in c.then.qubits}
        if default is not None:
            qubits |= set(default.qubits)
        self._qubits = tuple(sorted(qubits))

    @property
    def control(self) -> cirq.MeasurementKey:
        """The expression over measurement keys that selects the branch."""
        return self._control

    @property
    def cases(self) -> tuple[Case, ...]:
        """The branches, in insertion order."""
        return self._cases

    @property
    def default(self) -> raw_types.Operation | None:
        """The operation run when no case matches, if any."""
        return self._default

    def _branches(self) -> list[raw_types.Operation]:
        ops = [c.then for c in self._cases]
        return ops if self._default is None else [*ops, self._default]

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> Switch:
        old = self.qubits
        if len(new_qubits) != len(old):
            raise ValueError(f'Expected {len(old)} qubits, got {len(new_qubits)}.')
        qmap = dict(zip(old, new_qubits))
        new_cases = []
        for c in self.cases:
            qubit_mapping = [qmap[q] for q in c.then.qubits]
            new_sub = c.then.with_qubits(*qubit_mapping)
            new_cases.append(Case(c.values, new_sub))
        new_default = None
        if self.default is not None:
            qubit_mapping = [qmap[q] for q in self.default.qubits]
            new_default = self.default.with_qubits(*qubit_mapping)
        return Switch(self.control, new_cases, new_default)

    def without_classical_controls(self) -> raw_types.Operation:
        from cirq.circuits import CircuitOperation, FrozenCircuit

        if len(self._branches()) == 1:
            return self._branches()[0]

        # Returns all operations in sequence from cases, then the default if exists.
        seq = []
        for branch in self._branches():
            branch_no_control = branch.without_classical_controls()
            if isinstance(branch_no_control, CircuitOperation):
                seq.extend(protocols.decompose(branch_no_control))
            else:
                seq.append(branch_no_control)
        return CircuitOperation(FrozenCircuit(seq))

    @property
    def classical_controls(self) -> frozenset[cirq.Condition]:
        # Some code paths in Cirq assume sub operations are run when _all_ of the classical control
        # conditions are met. For `Switch` this is not the case; it has multiple classical controls,
        # each of which gate separate operations. Therefore, it's ambiguous as to how this method
        # should be implemented.
        return NotImplemented  # pragma: no cover

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        control_keys = {self.control}
        for branch in self._branches():
            control_keys.update(protocols.control_keys(branch))
        return frozenset(control_keys)

    def _decompose_(self) -> cirq.OP_TREE | NotImplementedType:
        case_conditions = []
        decomposed = []
        for c in self.cases:
            case_condition = c.get_condition(self.control)
            case_conditions.append(case_condition)
            decomposed.append(c.then.with_classical_controls(case_condition))
        if self.default is not None:
            # Implement the "default" condition as "none of the case conditions".
            decomposed.append(
                self.default.with_classical_controls(
                    value.OrCondition(tuple(case_conditions), invert=True)
                )
            )
        return decomposed

    def _act_on_(self, sim_state: cirq.SimulationStateBase) -> bool:
        acted = False
        for c in self.cases:
            if c.check(self.control, sim_state.classical_data):
                protocols.act_on(c.then, sim_state)
                acted = True
        if not acted and self._default is not None:
            protocols.act_on(self._default, sim_state)
        return True

    @_compat.cached_method
    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(c) for c in self._branches())

    @_compat.cached_method
    def _parameter_names_(self) -> frozenset[str]:
        name_sets = [protocols.parameter_names(c) for c in self.cases]
        if self.default is not None:
            name_sets.append(protocols.parameter_names(self.default))
        return set.union(*name_sets)

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> Switch:
        new_cases = [protocols.resolve_parameters(c, resolver, recursive) for c in self.cases]
        new_default = (
            protocols.resolve_parameters(self.default, resolver, recursive)
            if self.default is not None
            else None
        )
        return Switch(self.control, new_cases, new_default)

    def _value_equality_values_(self) -> Any:
        return self._control, self._cases, self._default

    def _json_dict_(self) -> dict[str, Any]:
        return {'control': self._control, 'cases': self._cases, 'default': self._default}

    @classmethod
    def _from_json_dict_(cls, control, cases, default=None, **kwargs) -> Switch:
        return cls(control, cases, default)

    def _draw_circuit_diagram(self, args: cirq.CircuitDiagramInfoArgs | None = None) -> str:
        from cirq.circuits import Circuit, CircuitOperation

        """Draws a circuit diagram.

        The diagram is represented as a sequence of circuit diagrams representing the case and
        default branches, for example:

        ```
        Switch(cond, | Case (2,):     | Case (0, 1):       | Default:   |
        ...          | 2: ───X───Z─── | 2: ─────────────── | 2: ─────── |
        ...          |                |                    |            |
        ...          | 3: ─────────── | 3: ───────X───Z─── | 3: ─────── |
        ...          |                |               ║    |            |
        ...          | 4: ─────────── | 4: ───────────╫─── | 4: ───X─── |
        ...          |                |               ║    |            |
        ...          |                | cond2: ═══════^═══ |            |)
        ```

        Args:
            args: An optional `cirq.CircuitDiagramInfoArgs` object for diagram info, if applicable.
        Returns:
            A string representing the circuit diagram.
        """

        # Each branch may make use of a different amount of qubits, and may require special
        # formatting which may or may not interact well with each other. To simplify things,
        # just turn everything in circuits and display branches sequentially.
        circuit_diagram_qubits = args.known_qubits if args is not None else self.qubits
        circuit_headers = []
        circuit_displays = []
        for c in self.cases:
            case_circuit = (
                c.then.circuit if isinstance(c.then, CircuitOperation) else Circuit(c.then)
            )
            case_circuit_display = case_circuit.to_text_diagram(qubit_order=circuit_diagram_qubits)
            circuit_headers.append(f'Case {c.values}:')
            circuit_displays.append(case_circuit_display.split('\n'))
        if self.default is not None:
            default_circuit = (
                self.default.circuit
                if isinstance(self.default, CircuitOperation)
                else Circuit(self.default)
            )
            default_circuit_display = default_circuit.to_text_diagram(
                qubit_order=circuit_diagram_qubits
            )
            circuit_headers.append('Default:')
            circuit_displays.append(default_circuit_display.split('\n'))
        lines = [''] + [
            '...' for _ in range(max(len(circuit_display) for circuit_display in circuit_displays))
        ]

        # Draw the "Switch" header
        switch_line = f'Switch({self.control.name}, '
        lines[0] = switch_line
        for idx in range(len(lines)):
            lines[idx] = lines[idx].ljust(len(switch_line)) + '|'

        for circuit_idx, (circuit_header, circuit_display) in enumerate(
            zip(circuit_headers, circuit_displays)
        ):
            circuit_lines = [circuit_header, *circuit_display]
            max_len = max(len(line) for line in circuit_lines)
            circuit_lines = [line.ljust(max_len) for line in circuit_lines]
            for idx in range(len(lines)):
                lines[idx] += (
                    ' '
                    + (circuit_lines[idx] if idx < len(circuit_lines) else ''.ljust(max_len))
                    + ' |'
                )

        lines[-1] += ')'
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self._draw_circuit_diagram()

    def __repr__(self) -> str:
        cases = ','.join([f'\n    {repr(c)}' for c in self.cases]).replace('\n', '\n    ')
        default = '' if self._default is None else f'\n    default={self._default!r}'
        return f'cirq.Switch({self._control!r}, ({cases}\n    ),{default}\n)'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> tuple[str, ...]:
        circuit_diagram_qubits = args.known_qubits
        if len(circuit_diagram_qubits) == 0:
            return ()  # pragma: no cover

        circuit_diagram = self._draw_circuit_diagram(args)
        other_lines = tuple(f'#{idx + 1}' for idx in range(1, len(circuit_diagram_qubits)))
        empty_measure_lines = tuple('' for _ in self._control_keys_())

        return (circuit_diagram, *other_lines, *empty_measure_lines)

    def _qasm_(
        self, *, args: cirq.QasmArgs | None = None, qubits: Sequence[cirq.Qid] | None = None
    ) -> str | None:
        if args is None:
            from cirq.protocols.qasm import QasmArgs

            args = QasmArgs()

        args.validate_version('3.1')
        return self.qasm(args=args, qubits=qubits)

    def qasm(self, *, args: cirq.QasmArgs, qubits: Sequence[cirq.Qid] | None = None) -> str | None:
        """Returns the QASM for this `Switch`, bypassing the QASM version check.

        `switch` is only part of OpenQASM 3.1 and later. This method allows emitting it for
        other versions as well.
        """
        control_expr = args.meas_key_id_map[self.control]
        cases_exprs = [c.qasm(args=args, qubits=qubits) for c in self.cases]
        default_expr = protocols.qasm(self.default, args=args, qubits=qubits, default=None)
        if any(case_expr is None for case_expr in cases_exprs):
            return None  # pragma: no cover

        def extend(text: str, spaces: int) -> str:
            padding = ' ' * spaces
            return ''.join(f'{padding}{line}\n' for line in text.split('\n'))

        cases_expr = '\n'.join(cases_exprs)
        if self.default is not None:
            if default_expr is None:
                return None
            default_expr = ''.join(f'\n    {line}' for line in default_expr.strip().split('\n'))
            default_expr = f'default {{{default_expr}\n}}'
            return f'switch ({control_expr}) {{\n{extend(cases_expr, 4)}{extend(default_expr, 4)}}}'

        return f'switch ({control_expr}) {{\n{extend(cases_expr, 4)}}}'
