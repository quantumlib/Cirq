# Copyright 2019 The Cirq Developers
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
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np

from cirq import protocols, qis, value
from cirq.ops import (
    controlled_gate,
    common_gates,
    eigen_gate,
    gate_operation,
    matrix_gates,
    op_tree,
    raw_types,
    control_values as cv,
)
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


@value.value_equality
class ControlledOperation(raw_types.Operation):
    """Augments existing operations to have one or more control qubits.

    This object is typically created via `operation.controlled_by(*qubits)`.
    """

    def __init__(
        self,
        controls: Sequence['cirq.Qid'],
        sub_operation: 'cirq.Operation',
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
    ):
        """Initializes the controlled operation.

        Args:
            controls: The qubits that control the sub-operation.
            sub_operation: The operation that will be controlled.
            control_values: Which control qubit values to apply the sub
                operation.  Either an object that inherits from AbstractControlValues
                or a sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                qubit value (or set of possible values) where that control is
                enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.

        Raises:
            ValueError: If the `control_values` or `control_qid_shape` does not
                match the number of qubits, if the `control_values` are out of
                bounds, if the qubits overlap, or if the sub_operation is not a
                unitary or mixture.
        """
        controlled_gate._validate_sub_object(sub_operation)
        if control_values is None:
            control_values = ((1,),) * len(controls)
        if not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(control_values)

        if protocols.num_qubits(control_values) != len(controls):
            raise ValueError('cirq.num_qubits(control_values) != len(controls)')

        # Verify qubits control qubits unique
        control_set = set(controls)
        if len(controls) != len(control_set):
            seen = set()
            dupes = [x for x in controls if x in seen or seen.add(x)]  # type: ignore
            raise ValueError(f'Duplicate control qubits {[str(x) for x in dupes]}.')

        # Verify qubits don't overlap
        if not control_set.isdisjoint(sub_operation.qubits):
            overlap = control_set.intersection(sub_operation.qubits)
            raise ValueError(f'Sub-op and controls share qubits {[str(x) for x in overlap]}.')

        self._control_values = control_values

        # Verify control values not out of bounds
        self._control_values.validate([q.dimension for q in controls])

        if not isinstance(sub_operation, ControlledOperation):
            self._controls = tuple(controls)
            self._sub_operation = sub_operation
        else:
            # Auto-flatten nested controlled operations.
            self._controls = tuple(controls) + sub_operation.controls
            self._sub_operation = sub_operation.sub_operation
            self._control_values = self._control_values & sub_operation.control_values

    @property
    def controls(self) -> Tuple['cirq.Qid', ...]:
        return self._controls

    @property
    def control_values(self) -> cv.AbstractControlValues:
        return self._control_values

    @property
    def sub_operation(self) -> 'cirq.Operation':
        return self._sub_operation

    @property
    def gate(self) -> Optional['cirq.ControlledGate']:
        if self.sub_operation.gate is None:
            return None
        return controlled_gate.ControlledGate(
            self.sub_operation.gate,
            control_values=self.control_values,
            control_qid_shape=[q.dimension for q in self.controls],
        )

    @property
    def qubits(self):
        return self.controls + self.sub_operation.qubits

    def with_qubits(self, *new_qubits):
        n = len(self.controls)
        return ControlledOperation(
            new_qubits[:n], self.sub_operation.with_qubits(*new_qubits[n:]), self.control_values
        )

    def _decompose_(self):
        return self._decompose_with_context_()

    def _decompose_with_context_(self, context: Optional['cirq.DecompositionContext'] = None):
        result = protocols.decompose_once_with_qubits(
            self.gate, self.qubits, NotImplemented, flatten=False, context=context
        )
        if result is not NotImplemented:
            return result

        if isinstance(self.sub_operation.gate, matrix_gates.MatrixGate):
            # Default decompositions of 2/3 qubit `cirq.MatrixGate` ignores global phase, which is
            # local phase in the controlled variant and hence cannot be ignored.
            return NotImplemented

        result = protocols.decompose_once(
            self.sub_operation, NotImplemented, flatten=False, context=context
        )
        if result is NotImplemented:
            return NotImplemented

        return op_tree.transform_op_tree(
            result, lambda op: op.controlled_by(*self.controls, control_values=self.control_values)
        )

    def _value_equality_values_(self):
        sorted_controls, expanded_cvals = tuple(
            zip(*sorted(zip(self.controls, zip(*self.control_values.expand()))))
        )
        return sorted_controls, tuple(expanded_cvals), self.sub_operation

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        n = len(self.controls)
        sub_n = len(args.axes) - n
        sub_axes = args.axes[n:]
        for control_vals in self.control_values.expand():
            active: Tuple[Union['ellipsis', slice], ...] = (
                ...,
                *(slice(v, v + 1) for v in control_vals),
                *(slice(None),) * sub_n,
            )
            target_view = args.target_tensor[active]
            buffer_view = args.available_buffer[active]
            result = protocols.apply_unitary(
                self.sub_operation,
                protocols.ApplyUnitaryArgs(target_view, buffer_view, sub_axes),
                default=NotImplemented,
            )

            if result is NotImplemented:
                return NotImplemented

            if result is not target_view:
                # HACK: assume they didn't somehow escape the slice view and
                # edit the rest of target_tensor.
                target_view[...] = result

        return args.target_tensor

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_operation)

    def _qasm_(self, args: 'cirq.QasmArgs') -> Optional[str]:
        if (
            hasattr(self._sub_operation, "gate")
            and len(self._controls) == 1
            and self.control_values == cv.ProductOfSums(((1,),))
        ):
            gate = self.sub_operation.gate
            if (
                isinstance(gate, eigen_gate.EigenGate)
                and gate.exponent == 1
                and gate.global_shift == 0
            ):
                instr = None
                if isinstance(gate, common_gates.XPowGate):
                    instr = 'cx {0},{1};\n'
                elif isinstance(gate, common_gates.YPowGate):
                    instr = 'cy {0},{1};\n'
                elif isinstance(gate, common_gates.ZPowGate):
                    instr = 'cz {0},{1};\n'
                elif isinstance(gate, common_gates.HPowGate):
                    instr = 'ch {0},{1};\n'
                if instr:
                    return args.format(instr, self._controls[0], self.sub_operation.qubits[0])
        # Fallback to decompose.
        return None

    def _extend_matrix(self, sub_matrix: np.ndarray) -> np.ndarray:
        qid_shape = protocols.qid_shape(self)
        sub_n = len(qid_shape) - len(self.controls)
        tensor = qis.eye_tensor(qid_shape, dtype=sub_matrix.dtype)
        sub_tensor = sub_matrix.reshape(qid_shape[len(self.controls) :] * 2)
        for control_vals in self.control_values.expand():
            active = (*(v for v in control_vals), *(slice(None),) * sub_n) * 2
            tensor[active] = sub_tensor
        return tensor.reshape((np.prod(qid_shape, dtype=np.int64).item(),) * 2)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        sub_matrix = protocols.unitary(self.sub_operation, None)
        if sub_matrix is None:
            return NotImplemented
        return self._extend_matrix(sub_matrix)

    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.sub_operation)

    def _mixture_(self) -> Optional[List[Tuple[float, np.ndarray]]]:
        sub_mixture = protocols.mixture(self.sub_operation, None)
        if sub_mixture is None:
            return None
        return [(p, self._extend_matrix(m)) for p, m in sub_mixture]

    def __str__(self) -> str:
        prefix = str(self.control_values)
        if isinstance(self.sub_operation, gate_operation.GateOperation):
            qubits = ', '.join(map(str, self.qubits))
            return f'{prefix}{self.sub_operation.gate}({qubits})'
        controls = ', '.join(str(q) for q in self.controls)
        return f'{prefix}({controls}, {self.sub_operation})'

    def __repr__(self):
        if all(q.dimension == 2 for q in self.controls):
            if self.control_values.is_trivial:
                if self == self.sub_operation.controlled_by(*self.controls):
                    qubit_args = ', '.join(repr(q) for q in self.controls)
                    return f'{self.sub_operation!r}.controlled_by({qubit_args})'
        return (
            f'cirq.ControlledOperation('
            f'sub_operation={self.sub_operation!r},'
            f'control_values={self.control_values!r},'
            f'controls={self.controls!r})'
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.sub_operation)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.sub_operation)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ControlledOperation':
        new_sub_op = protocols.resolve_parameters(self.sub_operation, resolver, recursive)
        return ControlledOperation(self.controls, new_sub_op, self.control_values)

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        u = protocols.unitary(self.sub_operation, default=None)
        if u is None:
            return NotImplemented
        angle_list = np.append(np.angle(np.linalg.eigvals(u)), 0)
        return protocols.trace_distance_from_angle_list(angle_list)

    def __pow__(self, exponent: Any) -> 'ControlledOperation':
        new_sub_op = protocols.pow(self.sub_operation, exponent, NotImplemented)
        if new_sub_op is NotImplemented:
            return NotImplemented
        return ControlledOperation(self.controls, new_sub_op, self.control_values)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Optional['protocols.CircuitDiagramInfo']:
        n = len(self.controls)

        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=(
                args.known_qubit_count - n if args.known_qubit_count is not None else None
            ),
            known_qubits=(args.known_qubits[n:] if args.known_qubits is not None else None),
            use_unicode_characters=args.use_unicode_characters,
            precision=args.precision,
            label_map=args.label_map,
        )
        sub_info = protocols.circuit_diagram_info(self.sub_operation, sub_args, None)
        if sub_info is None:
            return NotImplemented

        cv_info = protocols.circuit_diagram_info(self.control_values)

        wire_symbols = (*cv_info.wire_symbols, *sub_info.wire_symbols)
        exponent_qubit_index = None
        num_controls = protocols.num_qubits(self.control_values)
        if sub_info.exponent_qubit_index is not None:
            exponent_qubit_index = sub_info.exponent_qubit_index + num_controls
        elif sub_info.exponent is not None:
            # For a multi-qubit `sub_operation`, if the `exponent_qubit_index` is None, the qubit
            # on which the exponent gets drawn in the controlled case (smallest ordered qubit of
            # sub_operation) can be different from the uncontrolled case (lexicographically largest
            # qubit of sub_operation). See tests for example.
            exponent_qubit_index = num_controls
        return protocols.CircuitDiagramInfo(
            wire_symbols=wire_symbols,
            exponent=sub_info.exponent,
            exponent_qubit_index=exponent_qubit_index,
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'controls': self.controls,
            'control_values': self.control_values,
            'sub_operation': self.sub_operation,
        }
