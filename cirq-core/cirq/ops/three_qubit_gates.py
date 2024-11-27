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

"""Common quantum gates that target three qubits."""

from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import sympy

from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
    common_gates,
    controlled_gate,
    eigen_gate,
    gate_features,
    pauli_gates,
    raw_types,
    swap_gates,
    raw_types,
    control_values as cv,
    global_phase_op,
)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


class CCZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""A doubly-controlled-Z that can be raised to a power.

    The unitary matrix of `CCZ**t` is (empty elements are $0$):

    $$
    \begin{bmatrix}
        1 & & & & & & & \\
        & 1 & & & & & & \\
        & & 1 & & & & & \\
        & & & 1 & & & & \\
        & & & & 1 & & & \\
        & & & & & 1 & & \\
        & & & & & & 1 & \\
        & & & & & & & e^{i \pi t}
    \end{bmatrix}
    $$
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 1, 1, 1, 1, 1, 1, 0])), (1, np.diag([0, 0, 0, 0, 0, 0, 0, 1]))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j**self._exponent
        c = -1j * z_phase * np.sin(np.pi * self._exponent / 2) / 4
        return value.LinearDict(
            {
                'III': global_phase * (1 - c),
                'IIZ': global_phase * c,
                'IZI': global_phase * c,
                'ZII': global_phase * c,
                'ZZI': global_phase * -c,
                'ZIZ': global_phase * -c,
                'IZZ': global_phase * -c,
                'ZZZ': global_phase * c,
            }
        )

    def _decompose_(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───p───@──────────────@───────@──────────@──────────
                  │              │       │          │
        1: ───p───X───@───p^-1───X───@───X──────@───X──────@───
                      │              │          │          │
        2: ───p───────X───p──────────X───p^-1───X───p^-1───X───

        where p = T**self._exponent
        """
        a, b, c = qubits

        # Hacky magic: avoid the non-adjacent edge.
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = c, b
            elif not b.is_adjacent(c):
                a, b = b, a

        p = common_gates.T**self._exponent
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
        global_phase = 1j ** (2 * self.global_shift * self._exponent)
        global_phase = (
            complex(global_phase)
            if protocols.is_parameterized(global_phase) and global_phase.is_complex
            else global_phase
        )
        global_phase_operation = (
            [global_phase_op.global_phase_operation(global_phase)]
            if protocols.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0
            else []
        )
        return global_phase_operation + [
            p(a),
            p(b),
            p(c),
            sweep_abc,
            p(b) ** -1,
            p(c),
            sweep_abc,
            p(c) ** -1,
            sweep_abc,
            p(c) ** -1,
            sweep_abc,
        ]

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        ooo = args.subspace_index(0b111)
        args.target_tensor[ooo] *= np.exp(1j * self.exponent * np.pi)
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(('@', '@', '@'), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None

        args.validate_version('2.0', '3.0')
        lines = [
            args.format('h {0};\n', qubits[2]),
            args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2]),
            args.format('h {0};\n', qubits[2]),
        ]
        return ''.join(lines)

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CCZ'
            return f'(cirq.CCZ**{proper_repr(self._exponent)})'
        return (
            f'cirq.CCZPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CCZ'
        return f'CCZ**{self._exponent}'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `ZPowGate` with two additional controls.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = ZPowGate`.
        """
        if num_controls == 0:
            return self
        sub_gate: 'cirq.Gate' = self
        if self._global_shift == 0:
            sub_gate = controlled_gate.ControlledGate(
                common_gates.ZPowGate(exponent=self._exponent), num_controls=2
            )
        return controlled_gate.ControlledGate(
            sub_gate,
            num_controls=num_controls,
            control_values=control_values,
            control_qid_shape=control_qid_shape,
        )


@value.value_equality()
class ThreeQubitDiagonalGate(raw_types.Gate):
    r"""A three qubit gate whose unitary is given by a diagonal $8 \times 8$ matrix.

    This gate's off-diagonal elements are zero and its on diagonal
    elements are all phases.
    """

    def __init__(self, diag_angles_radians: Sequence[value.TParamVal]) -> None:
        r"""A three qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and its on diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \ldots , x_7)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \ldots, e^{i x_7})$.
        """
        self._diag_angles_radians: Tuple[value.TParamVal, ...] = tuple(diag_angles_radians)

    @property
    def diag_angles_radians(self) -> Tuple[value.TParamVal, ...]:
        return self._diag_angles_radians

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(angle) for angle in self._diag_angles_radians)

    def _parameter_names_(self) -> AbstractSet[str]:
        return {
            name for angle in self._diag_angles_radians for name in protocols.parameter_names(angle)
        }

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'ThreeQubitDiagonalGate':
        return self.__class__(
            [
                protocols.resolve_parameters(angle, resolver, recursive)
                for angle in self._diag_angles_radians
            ]
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        return np.diag([np.exp(1j * angle) for angle in self._diag_angles_radians])

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        for index, angle in enumerate(self._diag_angles_radians):
            little_endian_index = 4 * (index & 1) + 2 * ((index >> 1) & 1) + ((index >> 2) & 1)
            subspace_index = args.subspace_index(little_endian_index)
            args.target_tensor[subspace_index] *= np.exp(1j * angle)
        return args.target_tensor

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        rounded_angles = np.array(self._diag_angles_radians)
        if args.precision is not None:
            rounded_angles = rounded_angles.round(args.precision)
        diag_str = f"diag({', '.join(proper_repr(angle) for angle in rounded_angles)})"
        return protocols.CircuitDiagramInfo((diag_str, '#2', '#3'))

    def __pow__(self, exponent: Any) -> 'ThreeQubitDiagonalGate':
        if not isinstance(exponent, (int, float, sympy.Basic)):
            return NotImplemented
        return ThreeQubitDiagonalGate(
            [protocols.mul(angle, exponent, NotImplemented) for angle in self._diag_angles_radians]
        )

    def _decompose_(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───p_0───@──────────────@───────@──────────@──────────
                    │              │       │          │
        1: ───p_1───X───@───p_3────X───@───X──────@───X──────@───
                        │              │          │          │
        2: ───p_2───────X───p_4────────X───p_5────X───p_6────X───

        where p_i = T**(4*x_i) and x_i solve the system of equations
                    [0, 0, 1, 0, 1, 1, 1][x_0]   [r_1]
                    [0, 1, 0, 1, 1, 0, 1][x_1]   [r_2]
                    [0, 1, 1, 1, 0, 1, 0][x_2]   [r_3]
                    [1, 0, 0, 1, 1, 1, 0][x_3] = [r_4]
                    [1, 0, 1, 1, 0, 0, 1][x_4]   [r_5]
                    [1, 1, 0, 0, 0, 1, 1][x_5]   [r_6]
                    [1, 1, 1, 0, 1, 0, 0][x_6]   [r_7]
        where r_i is self._diag_angles_radians[i].

        The above system was created by equating the composition of the gates
        in the circuit diagram to np.diag(self._diag_angles) (shifted by a
        global phase of np.exp(-1j * self._diag_angles[0])).
        """

        a, b, c = qubits
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = c, b
            elif not b.is_adjacent(c):
                a, b = b, a
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
        phase_matrix_inverse = 0.25 * np.array(
            [
                [-1, -1, -1, 1, 1, 1, 1],
                [-1, 1, 1, -1, -1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1],
                [-1, 1, 1, 1, 1, -1, -1],
                [1, 1, -1, 1, -1, -1, 1],
                [1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1],
            ]
        )
        shifted_angles_tail = [
            angle - self._diag_angles_radians[0] for angle in self._diag_angles_radians[1:]
        ]
        phase_solutions = phase_matrix_inverse.dot(shifted_angles_tail)
        p_gates = [pauli_gates.Z ** (solution / np.pi) for solution in phase_solutions]
        global_phase = 1j ** (2 * self._diag_angles_radians[0] / np.pi)
        global_phase_operation = (
            [global_phase_op.global_phase_operation(global_phase)]
            if protocols.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0
            else []
        )
        return global_phase_operation + [
            p_gates[0](a),
            p_gates[1](b),
            p_gates[2](c),
            sweep_abc,
            p_gates[3](b),
            p_gates[4](c),
            sweep_abc,
            p_gates[5](c),
            sweep_abc,
            p_gates[6](c),
            sweep_abc,
        ]

    def _value_equality_values_(self):
        return tuple(self._diag_angles_radians)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        x = [np.exp(1j * angle) for angle in self._diag_angles_radians]
        return value.LinearDict(
            {
                'III': (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]) / 8,
                'IIZ': (x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7]) / 8,
                'IZI': (x[0] + x[1] - x[2] - x[3] + x[4] + x[5] - x[6] - x[7]) / 8,
                'IZZ': (x[0] - x[1] - x[2] + x[3] + x[4] - x[5] - x[6] + x[7]) / 8,
                'ZII': (x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7]) / 8,
                'ZIZ': (x[0] - x[1] + x[2] - x[3] - x[4] + x[5] - x[6] + x[7]) / 8,
                'ZZI': (x[0] + x[1] - x[2] - x[3] - x[4] - x[5] + x[6] + x[7]) / 8,
                'ZZZ': (x[0] - x[1] - x[2] + x[3] - x[4] + x[5] + x[6] - x[7]) / 8,
            }
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=["diag_angles_radians"])

    def __repr__(self) -> str:
        angles = ','.join(proper_repr(angle) for angle in self._diag_angles_radians)
        return f'cirq.ThreeQubitDiagonalGate([{angles}])'

    def _num_qubits_(self) -> int:
        return 3


class CCXPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""A Toffoli (doubly-controlled-NOT) that can be raised to a power.

    The unitary matrix of `CCX**t` is an 8x8 identity except the bottom right
    2x2 area is the matrix of `X**t`:

    $$
    \begin{bmatrix}
        1 & & & & & & & \\
        & 1 & & & & & & \\
        & & 1 & & & & & \\
        & & & 1 & & & & \\
        & & & & 1 & & & \\
        & & & & & 1 & & \\
        & & & & & & e^{i \pi t /2} \cos(\pi t) & -i e^{i \pi t /2} \sin(\pi t) \\
        & & & & & & -i e^{i \pi t /2} \sin(\pi t) & e^{i \pi t /2} \cos(\pi t)
    \end{bmatrix}
    $$
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, linalg.block_diag(np.diag([1, 1, 1, 1, 1, 1]), np.array([[0.5, 0.5], [0.5, 0.5]]))),
            (
                1,
                linalg.block_diag(
                    np.diag([0, 0, 0, 0, 0, 0]), np.array([[0.5, -0.5], [-0.5, 0.5]])
                ),
            ),
        ]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j**self._exponent
        c = -1j * z_phase * np.sin(np.pi * self._exponent / 2) / 4
        return value.LinearDict(
            {
                'III': global_phase * (1 - c),
                'IIX': global_phase * c,
                'IZI': global_phase * c,
                'ZII': global_phase * c,
                'ZZI': global_phase * -c,
                'ZIX': global_phase * -c,
                'IZX': global_phase * -c,
                'ZZX': global_phase * c,
            }
        )

    def qubit_index_to_equivalence_group_key(self, index):
        return index < 2

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return protocols.apply_unitary(
            controlled_gate.ControlledGate(
                controlled_gate.ControlledGate(pauli_gates.X**self.exponent)
            ),
            protocols.ApplyUnitaryArgs(args.target_tensor, args.available_buffer, args.axes),
            default=NotImplemented,
        )

    def _decompose_(self, qubits):
        c1, c2, t = qubits
        yield common_gates.H(t)
        yield CCZPowGate(exponent=self._exponent, global_shift=self.global_shift).on(c1, c2, t)
        yield common_gates.H(t)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            ('@', '@', 'X'), exponent=self._diagram_exponent(args), exponent_qubit_index=2
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None

        args.validate_version('2.0', '3.0')
        return args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.TOFFOLI'
            return f'(cirq.TOFFOLI**{proper_repr(self._exponent)})'
        return (
            f'cirq.CCXPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'TOFFOLI'
        return f'TOFFOLI**{self._exponent}'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `XPowGate` with two additional controls.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = XPowGate`.
        """
        if num_controls == 0:
            return self
        sub_gate: 'cirq.Gate' = self
        if self._global_shift == 0:
            sub_gate = controlled_gate.ControlledGate(
                common_gates.XPowGate(exponent=self._exponent), num_controls=2
            )
        return controlled_gate.ControlledGate(
            sub_gate,
            num_controls=num_controls,
            control_values=control_values,
            control_qid_shape=control_qid_shape,
        )


@value.value_equality()
class CSwapGate(gate_features.InterchangeableQubitsGate, raw_types.Gate):
    """A controlled swap gate. The Fredkin gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index == 0 else 1

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return value.LinearDict(
            {
                'III': 3 / 4,
                'IXX': 1 / 4,
                'IYY': 1 / 4,
                'IZZ': 1 / 4,
                'ZII': 1 / 4,
                'ZXX': -1 / 4,
                'ZYY': -1 / 4,
                'ZZZ': -1 / 4,
            }
        )

    def _trace_distance_bound_(self) -> float:
        return 1.0

    def _decompose_(self, qubits):
        c, t1, t2 = qubits

        # Hacky magic: special case based on adjacency.
        if hasattr(t1, 'is_adjacent'):
            if not t1.is_adjacent(t2):
                # Targets separated by control.
                return self._decompose_inside_control(t1, c, t2)
            if not t1.is_adjacent(c):
                # Control separated from t1 by t2.
                return self._decompose_outside_control(c, t2, t1)

        return self._decompose_outside_control(c, t1, t2)

    def _decompose_inside_control(
        self, target1: 'cirq.Qid', control: 'cirq.Qid', target2: 'cirq.Qid'
    ) -> Iterator['cirq.OP_TREE']:
        """A decomposition assuming the control separates the targets.

        target1: ─@─X───────T──────@────────@─────────X───@─────X^-0.5─
                  │ │              │        │         │   │
        control: ─X─@─X─────@─T^-1─X─@─T────X─@─X^0.5─@─@─X─@──────────
                      │     │        │        │         │   │
        target2: ─────@─H─T─X─T──────X─T^-1───X─T^-1────X───X─H─S^-1───
        """
        a, b, c = target1, control, target2
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(c, b)
        yield common_gates.H(c)
        yield common_gates.T(c)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(a)
        yield common_gates.T(b) ** -1
        yield common_gates.T(c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(b)
        yield common_gates.T(c) ** -1
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield pauli_gates.X(b) ** 0.5
        yield common_gates.T(c) ** -1
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(b, c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.H(c)
        yield common_gates.S(c) ** -1
        yield pauli_gates.X(a) ** -0.5

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        return protocols.apply_unitary(
            controlled_gate.ControlledGate(swap_gates.SWAP),
            protocols.ApplyUnitaryArgs(args.target_tensor, args.available_buffer, args.axes),
            default=NotImplemented,
        )

    def _decompose_outside_control(
        self, control: 'cirq.Qid', near_target: 'cirq.Qid', far_target: 'cirq.Qid'
    ) -> Iterator['cirq.OP_TREE']:
        """A decomposition assuming one of the targets is in the middle.

        control: ───T──────@────────@───@────────────@────────────────
                           │        │   │            │
           near: ─X─T──────X─@─T^-1─X─@─X────@─X^0.5─X─@─X^0.5────────
                  │          │        │      │         │
            far: ─@─Y^-0.5─T─X─T──────X─T^-1─X─T^-1────X─S─────X^-0.5─
        """
        a, b, c = control, near_target, far_target

        t = common_gates.T
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]

        yield common_gates.CNOT(c, b)
        yield pauli_gates.Y(c) ** -0.5
        yield t(a), t(b), t(c)
        yield sweep_abc
        yield t(b) ** -1, t(c)
        yield sweep_abc
        yield t(c) ** -1
        yield sweep_abc
        yield t(c) ** -1
        yield pauli_gates.X(b) ** 0.5
        yield sweep_abc
        yield common_gates.S(c)
        yield pauli_gates.X(b) ** 0.5
        yield pauli_gates.X(c) ** -0.5

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return linalg.block_diag(np.diag([1, 1, 1, 1, 1]), np.array([[0, 1], [1, 0]]), np.diag([1]))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(('@', 'swap', 'swap'))
        return protocols.CircuitDiagramInfo(('@', '×', '×'))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        return args.format('cswap {0},{1},{2};\n', qubits[0], qubits[1], qubits[2])

    def _value_equality_values_(self):
        return ()

    def __pow__(self, power):
        if power == 1 or power == -1:
            return self
        return NotImplemented

    def __str__(self) -> str:
        return 'FREDKIN'

    def __repr__(self) -> str:
        return 'cirq.FREDKIN'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> raw_types.Gate:
        """Returns a controlled `SWAP` with one additional control.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = SWAP`.
        """
        if num_controls == 0:
            return self
        return controlled_gate.ControlledGate(
            controlled_gate.ControlledGate(swap_gates.SWAP, num_controls=1),
            num_controls=num_controls,
            control_values=control_values,
            control_qid_shape=control_qid_shape,
        )


CCZ = CCZPowGate()
document(
    CCZ,
    r"""The Controlled-Controlled-Z gate.

    The `exponent=1` instance of `cirq.CCZPowGate`.

    The unitary matrix of this gate is (empty elements are $0$):
    $$
    \begin{bmatrix}
        1 & & & & & & & \\
        & 1 & & & & & & \\
        & & 1 & & & & & \\
        & & & 1 & & & & \\
        & & & & 1 & & & \\
        & & & & & 1 & & \\
        & & & & & & 1 & \\
        & & & & & & & -1
    \end{bmatrix}
    $$
    """,
)

CCNotPowGate = CCXPowGate
CCX = TOFFOLI = CCNOT = CCXPowGate()
document(
    CCX,
    r"""The Tofolli gate, also known as the Controlled-Controlled-X gate.

    If the first two qubits are in the |11⟩ state, this flips the third qubit
    in the computational basis, otherwise this applies identity to the third qubit.

    The `exponent=1` instance of `cirq.CCXPowGate`.

    The unitary matrix of this gate is (empty elements are $0$):

    $$
    \begin{bmatrix}
        1 & & & & & & & \\
        & 1 & & & & & & \\
        & & 1 & & & & & \\
        & & & 1 & & & & \\
        & & & & 1 & & & \\
        & & & & & 1 & & \\
        & & & & & & 0 & 1 \\
        & & & & & & 1 & 0
    \end{bmatrix}
    $$

    Alternative names: `cirq.CCNOT` and `cirq.TOFFOLI`.
    """,
)

CSWAP = FREDKIN = CSwapGate()
document(
    CSWAP,
    r"""The Controlled Swap gate, also known as the Fredkin gate.

    If the first qubit is |1⟩, this applies a SWAP between the second and third qubit,
    otherwise it acts as identity on the second and third qubit.

    An instance of `cirq.CSwapGate`.

    The unitary matrix of this gate is (empty elements are $0$):
    $$
    \begin{bmatrix}
        1 & & & & & & & \\
        & 1 & & & & & & \\
        & & 1 & & & & & \\
        & & & 1 & & & & \\
        & & & & 1 & & & \\
        & & & & & 0 & 1 & \\
        & & & & & 1 & 0 & \\
        & & & & & & & 1
    \end{bmatrix}
    $$

    Alternative names: `cirq.FREDKIN`.
    """,
)
