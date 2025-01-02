# Copyright 2018 The Cirq Developers
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

from types import NotImplementedType
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

from cirq import protocols, value, _import
from cirq.ops import (
    raw_types,
    controlled_operation as cop,
    op_tree,
    matrix_gates,
    control_values as cv,
)

if TYPE_CHECKING:
    import cirq

controlled_gate_decomposition = _import.LazyLoader(
    'controlled_gate_decomposition', globals(), 'cirq.transformers.analytical_decompositions'
)
common_gates = _import.LazyLoader('common_gates', globals(), 'cirq.ops')
line_qubit = _import.LazyLoader('line_qubit', globals(), 'cirq.devices')


@value.value_equality
class ControlledGate(raw_types.Gate):
    """Augments existing gates to have one or more control qubits.

    This object is typically created via `gate.controlled()`.
    """

    def __init__(
        self,
        sub_gate: 'cirq.Gate',
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Sequence[int]] = None,
    ) -> None:
        """Initializes the controlled gate. If no arguments are specified for
           the controls, defaults to a single qubit control.

        Args:
            sub_gate: The gate to add a control qubit to.
            num_controls: Total number of control qubits.
            control_values: For which control qubit values to apply the sub
                gate.  Either an object that inherits from AbstractControlValues
                or a sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                qubit value (or set of possible values) where that control is
                enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Raises:
            ValueError: If the `control_values` or `control_qid_shape` does not
                match with `num_controls`, if the `control_values` are out of
                bounds, or if the sub_gate is not a unitary or mixture.
        """
        _validate_sub_object(sub_gate)

        # Simplify a single SumOfProducts
        if isinstance(control_values, cv.SumOfProducts) and len(control_values._conjunctions) == 1:
            control_values = control_values._conjunctions[0]

        if num_controls is None:
            if control_values is not None:
                num_controls = (
                    control_values._num_qubits_()
                    if isinstance(control_values, cv.AbstractControlValues)
                    else len(control_values)
                )
            elif control_qid_shape is not None:
                num_controls = len(control_qid_shape)
            else:
                num_controls = 1
        if control_values is None:
            control_values = ((1,),) * num_controls

        # Convert to `cv.ProductOfSums` if input is a tuple of control values for each qubit.
        if not isinstance(control_values, cv.AbstractControlValues):
            control_values = cv.ProductOfSums(control_values)

        if num_controls != protocols.num_qubits(control_values):
            raise ValueError('cirq.num_qubits(control_values) != num_controls')

        if control_qid_shape is None:
            control_qid_shape = (2,) * num_controls
        if num_controls != len(control_qid_shape):
            raise ValueError('len(control_qid_shape) != num_controls')
        self._control_qid_shape = tuple(control_qid_shape)

        self._control_values = control_values

        # Verify control values not out of bounds
        self._control_values.validate(self.control_qid_shape)

        # Flatten nested ControlledGates.
        if isinstance(sub_gate, ControlledGate):
            self._sub_gate = sub_gate.sub_gate
            self._control_values = self._control_values & sub_gate.control_values
            self._control_qid_shape += sub_gate.control_qid_shape
        else:
            self._sub_gate = sub_gate

    @property
    def control_qid_shape(self) -> Tuple[int, ...]:
        return self._control_qid_shape

    @property
    def control_values(self) -> cv.AbstractControlValues:
        return self._control_values

    @property
    def sub_gate(self) -> 'cirq.Gate':
        return self._sub_gate

    def num_controls(self) -> int:
        return len(self.control_qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self.control_qid_shape + protocols.qid_shape(self.sub_gate)

    def _decompose_(
        self, qubits: Tuple['cirq.Qid', ...]
    ) -> Union[None, NotImplementedType, 'cirq.OP_TREE']:
        return self._decompose_with_context_(qubits)

    def _decompose_with_context_(
        self, qubits: Tuple['cirq.Qid', ...], context: Optional['cirq.DecompositionContext'] = None
    ) -> Union[None, NotImplementedType, 'cirq.OP_TREE']:
        if (
            protocols.has_unitary(self.sub_gate)
            and protocols.num_qubits(self.sub_gate) == 1
            and self._qid_shape_() == (2,) * len(self._qid_shape_())
            and isinstance(self.control_values, cv.ProductOfSums)
        ):
            control_qubits = list(qubits[: self.num_controls()])
            invert_ops: List['cirq.Operation'] = []
            for cvals, cqbit in zip(self.control_values, qubits[: self.num_controls()]):
                if set(cvals) == {0}:
                    invert_ops.append(common_gates.X(cqbit))
                elif set(cvals) == {0, 1}:
                    control_qubits.remove(cqbit)
            decomposed_ops = controlled_gate_decomposition.decompose_multi_controlled_rotation(
                protocols.unitary(self.sub_gate), control_qubits, qubits[-1]
            )
            return invert_ops + decomposed_ops + invert_ops

        if isinstance(self.sub_gate, common_gates.CZPowGate):
            z_sub_gate = common_gates.ZPowGate(
                exponent=self.sub_gate.exponent, global_shift=self.sub_gate.global_shift
            )
            num_controls = self.num_controls() + 1
            control_values = self.control_values & cv.ProductOfSums(((1,),))
            control_qid_shape = self.control_qid_shape + (2,)
            controlled_z = (
                z_sub_gate.controlled(
                    num_controls=num_controls,
                    control_values=control_values,
                    control_qid_shape=control_qid_shape,
                )
                if protocols.is_parameterized(self)
                else ControlledGate(
                    z_sub_gate,
                    num_controls=num_controls,
                    control_values=control_values,
                    control_qid_shape=control_qid_shape,
                )
            )
            if self != controlled_z:
                return protocols.decompose_once_with_qubits(
                    controlled_z, qubits, NotImplemented, context=context
                )

        if isinstance(self.sub_gate, matrix_gates.MatrixGate):
            # Default decompositions of 2/3 qubit `cirq.MatrixGate` ignores global phase, which is
            # local phase in the controlled variant and hence cannot be ignored.
            return NotImplemented

        result = protocols.decompose_once_with_qubits(
            self.sub_gate,
            qubits[self.num_controls() :],
            NotImplemented,
            flatten=False,
            context=context,
        )
        if result is NotImplemented:
            return NotImplemented

        return op_tree.transform_op_tree(
            result,
            lambda op: op.controlled_by(
                *qubits[: self.num_controls()], control_values=self.control_values
            ),
        )

    def on(self, *qubits: 'cirq.Qid') -> cop.ControlledOperation:
        if len(qubits) == 0:
            raise ValueError(f"Applied a gate to an empty set of qubits. Gate: {self!r}")
        self.validate_args(qubits)
        return cop.ControlledOperation(
            qubits[: self.num_controls()],
            self.sub_gate.on(*qubits[self.num_controls() :]),
            self.control_values,
        )

    def _value_equality_values_(self):
        return (self.sub_gate, self.num_controls(), self.control_values, self.control_qid_shape)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        qubits = line_qubit.LineQid.for_gate(self)
        op = self.sub_gate.on(*qubits[self.num_controls() :])
        c_op = cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)
        return protocols.apply_unitary(c_op, args, default=NotImplemented)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        qubits = line_qubit.LineQid.for_gate(self)
        op = self.sub_gate.on(*qubits[self.num_controls() :])
        c_op = cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)

        return protocols.unitary(c_op, default=NotImplemented)

    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.sub_gate)

    def _mixture_(self) -> Union[Sequence[tuple[float, np.ndarray]], NotImplementedType]:
        qubits = line_qubit.LineQid.for_gate(self)
        op = self.sub_gate.on(*qubits[self.num_controls() :])
        c_op = cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)
        return protocols.mixture(c_op, default=NotImplemented)

    def __pow__(self, exponent: Any) -> 'ControlledGate':
        new_sub_gate = protocols.pow(self.sub_gate, exponent, NotImplemented)
        if new_sub_gate is NotImplemented:
            return NotImplemented
        return ControlledGate(
            new_sub_gate,
            self.num_controls(),
            control_values=self.control_values,
            control_qid_shape=self.control_qid_shape,
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.sub_gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.sub_gate)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ControlledGate':
        new_sub_gate = protocols.resolve_parameters(self.sub_gate, resolver, recursive)
        return ControlledGate(
            new_sub_gate,
            self.num_controls(),
            control_values=self.control_values,
            control_qid_shape=self.control_qid_shape,
        )

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        u = protocols.unitary(self.sub_gate, default=None)
        if u is None:
            return NotImplemented
        angle_list = np.append(np.angle(np.linalg.eigvals(u)), 0)
        return protocols.trace_distance_from_angle_list(angle_list)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=(
                args.known_qubit_count - self.num_controls()
                if args.known_qubit_count is not None
                else None
            ),
            known_qubits=(
                args.known_qubits[self.num_controls() :] if args.known_qubits is not None else None
            ),
            use_unicode_characters=args.use_unicode_characters,
            precision=args.precision,
            label_map=args.label_map,
        )
        sub_info = protocols.circuit_diagram_info(self.sub_gate, sub_args, None)
        if sub_info is None:
            return NotImplemented

        cv_info = protocols.circuit_diagram_info(self.control_values)

        return protocols.CircuitDiagramInfo(
            wire_symbols=(*cv_info.wire_symbols, *sub_info.wire_symbols), exponent=sub_info.exponent
        )

    def __str__(self) -> str:
        return str(self.control_values) + str(self.sub_gate)

    def __repr__(self) -> str:
        if self.num_controls() == 1 and self.control_values.is_trivial:
            return f'cirq.ControlledGate(sub_gate={self.sub_gate!r})'

        if self.control_values.is_trivial and set(self.control_qid_shape) == {2}:
            return (
                f'cirq.ControlledGate(sub_gate={self.sub_gate!r}, '
                f'num_controls={self.num_controls()!r})'
            )
        return (
            f'cirq.ControlledGate(sub_gate={self.sub_gate!r}, '
            f'control_values={self.control_values!r},'
            f'control_qid_shape={self.control_qid_shape!r})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'control_values': self.control_values,
            'control_qid_shape': self.control_qid_shape,
            'sub_gate': self.sub_gate,
        }


def _validate_sub_object(sub_object: Union['cirq.Gate', 'cirq.Operation']):
    if protocols.is_measurement(sub_object):
        raise ValueError(f'Cannot control measurement {sub_object}')
    if not protocols.has_mixture(sub_object) and not protocols.is_parameterized(sub_object):
        raise ValueError(f'Cannot control channel with non-unitary operators: {sub_object}')
