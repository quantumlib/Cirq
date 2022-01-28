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

from typing import AbstractSet, Any, Collection, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import cirq
from cirq import protocols, value
from cirq.ops import raw_types, controlled_operation as cop, control_values as cv
from cirq.type_workarounds import NotImplementedType


@value.value_equality
class ControlledGate(raw_types.Gate):
    """Augments existing gates to have one or more control qubits.

    This object is typically created via `gate.controlled()`.
    """

    def __init__(
        self,
        sub_gate: 'cirq.Gate',
        num_controls: int = None,
        control_values: Optional[
            Union[cv.ControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Sequence[int]] = None,
    ) -> None:
        """Initializes the controlled gate. If no arguments are specified for
           the controls, defaults to a single qubit control.

        Args:
            sub_gate: The gate to add a control qubit to.
            num_controls: Total number of control qubits.
            control_values: For which control qubit values to apply the sub
                gate.  Either a ControlValues object or a sequence of length
                `num_controls` where each entry is an integer (or set of integers)
                corresponding to the qubit value (or set of possible values) where
                that control is enabled or a ControlValues object that specifies
                combinations of values for one or more qubits.
                When all controls are enabled, the sub gate is applied.
                If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Raises:
            ValueError: If the `control_values` or `control_qid_shape` does not
                match with `num_conrols`, or if the `control_values` are out of
                bounds.
        """
        if num_controls is None:
            if control_values is not None:
                num_controls = len(control_values)
            elif control_qid_shape is not None:
                num_controls = len(control_qid_shape)
            else:
                num_controls = 1
        if control_values is None:
            control_values = ((1,),) * num_controls
        if num_controls != len(control_values):
            raise ValueError('len(control_values) != num_controls')

        if control_qid_shape is None:
            control_qid_shape = (2,) * num_controls
        if num_controls != len(control_qid_shape):
            raise ValueError('len(control_qid_shape) != num_controls')
        self.control_qid_shape = tuple(control_qid_shape)

        self.control_values = cv.to_control_values(control_values)

        # Verify control values not out of bounds
        self.control_values.check_dimensionality(self.control_qid_shape)

        # Flatten nested ControlledGates.
        if isinstance(sub_gate, ControlledGate):
            self.sub_gate = sub_gate.sub_gate  # type: ignore
            self.control_values.product(sub_gate.control_values)
            self.control_qid_shape += sub_gate.control_qid_shape
        else:
            self.sub_gate = sub_gate

    @property
    def control_values(self) -> cv.ControlValues:
        return self._control_values

    @control_values.setter
    def control_values(
        self, values: Union[cv.ControlValues, Sequence[Union[int, Collection[int]]]]
    ) -> None:
        self._control_values = cv.to_control_values(values)

    def num_controls(self) -> int:
        return len(self.control_qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self.control_qid_shape + cirq.qid_shape(self.sub_gate)

    def _decompose_(self, qubits):
        result = protocols.decompose_once_with_qubits(
            self.sub_gate, qubits[self.num_controls() :], NotImplemented
        )

        if result is NotImplemented:
            return NotImplemented

        decomposed = []
        for op in result:
            decomposed.append(
                cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)
            )
        return decomposed

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
        return (
            self.sub_gate,
            self.num_controls(),
            frozenset(self.control_values.identifier(self.control_qid_shape)),
        )

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        qubits = cirq.LineQid.for_gate(self)
        op = self.sub_gate.on(*qubits[self.num_controls() :])
        c_op = cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)
        return protocols.apply_unitary(c_op, args, default=NotImplemented)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        qubits = cirq.LineQid.for_gate(self)
        op = self.sub_gate.on(*qubits[self.num_controls() :])
        c_op = cop.ControlledOperation(qubits[: self.num_controls()], op, self.control_values)

        return protocols.unitary(c_op, default=NotImplemented)

    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.sub_gate)

    def _mixture_(self) -> Union[np.ndarray, NotImplementedType]:
        qubits = cirq.LineQid.for_gate(self)
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

        def get_symbol(vals):
            if tuple(vals) == (1,):
                return '@'
            return f"({','.join(map(str, vals))})"

        return protocols.CircuitDiagramInfo(
            wire_symbols=(
                *(get_symbol(vals) for vals in self.control_values.arrangements()),
                *sub_info.wire_symbols,
            ),
            exponent=sub_info.exponent,
        )

    def __str__(self) -> str:
        if self.control_values.are_same_value(1):

            def get_prefix(control_vals):
                return 'C'

        else:

            def get_prefix(control_vals):
                control_vals_str = ''.join(map(str, sorted(control_vals)))
                return f'C{control_vals_str}'

        return ''.join(map(get_prefix, self.control_values.arrangements())) + str(self.sub_gate)

    def __repr__(self) -> str:
        if self.num_controls() == 1 and self.control_values.are_same_value(1):
            return f'cirq.ControlledGate(sub_gate={self.sub_gate!r})'

        if self.control_values.are_same_value(1) and set(self.control_qid_shape) == {2}:
            return (
                f'cirq.ControlledGate(sub_gate={self.sub_gate!r}, '
                f'num_controls={self.num_controls()!r})'
            )
        return (
            f'cirq.ControlledGate(sub_gate={self.sub_gate!r}, '
            f'control_values={self.control_values.arrangements()!r},'
            f'control_qid_shape={self.control_qid_shape!r})'
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'control_values': self.control_values.arrangements(),
            'control_qid_shape': self.control_qid_shape,
            'sub_gate': self.sub_gate,
        }
