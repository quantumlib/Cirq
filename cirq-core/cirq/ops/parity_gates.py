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

"""Quantum gates that phase with respect to product-of-pauli observables."""

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from typing_extensions import Self

import numpy as np

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import gate_features, eigen_gate, common_gates, pauli_gates

if TYPE_CHECKING:
    import cirq


@value.value_equality
class XXPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""The X-parity gate, possibly raised to a power.

    The XX**t gate implements the following unitary:

    $$
    (X \otimes X)^t = \begin{bmatrix}
                      c & 0 & 0 & s \\
                      0 & c & s & 0 \\
                      0 & s & c & 0 \\
                      s & 0 & 0 & c
                      \end{bmatrix}
    $$

    where

    $$
    c = f \cos\left(\frac{\pi t}{2}\right)
    $$

    $$
    s = -i f \sin\left(\frac{\pi t}{2}\right)
    $$

    $$
    f = e^{\frac{i \pi t}{2}}.
    $$

    See also: `cirq.ops.MSGate` (the Mølmer–Sørensen gate), which is
    implemented via this class.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (
                0.0,
                np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5]]),
            ),
            (
                1.0,
                np.array(
                    [[0.5, 0, 0, -0.5], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [-0.5, 0, 0, 0.5]]
                ),
            ),
        ]

    def _eigen_shifts(self):
        return [0, 1]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return [
                PauliInteractionGate(pauli_gates.X, False, pauli_gates.X, False).on(*qubits),
                SingleQubitCliffordGate.X_sqrt.on_each(*qubits),
            ]
        if self.exponent % 2 == 1:
            return [SingleQubitCliffordGate.X.on_each(*qubits)]
        if self.exponent % 2 == 1.5:
            return [
                PauliInteractionGate(pauli_gates.X, False, pauli_gates.X, False).on(*qubits),
                SingleQubitCliffordGate.X_nsqrt.on_each(*qubits),
            ]
        return NotImplemented

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> 'cirq.OP_TREE':
        yield common_gates.YPowGate(exponent=-0.5).on_each(*qubits)
        yield ZZPowGate(exponent=self.exponent, global_shift=self.global_shift)(*qubits)
        yield common_gates.YPowGate(exponent=0.5).on_each(*qubits)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX', 'XX'), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'XX'
        return f'XX**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.XX'
            return f'(cirq.XX**{proper_repr(self._exponent)})'
        return (
            f'cirq.XXPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


@value.value_equality
class YYPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""The Y-parity gate, possibly raised to a power.

    The YY**t gate implements the following unitary:

    $$
    (Y \otimes Y)^t = \begin{bmatrix}
                      c & 0 & 0 & -s \\
                      0 & c & s & 0 \\
                      0 & s & c & 0 \\
                      -s & 0 & 0 & c \\
                      \end{bmatrix}
    $$

    where

    $$
    c = f \cos\left(\frac{\pi t}{2}\right)
    $$

    $$
    s = -i f \sin\left(\frac{\pi t}{2}\right)
    $$

    $$
    f = e^{\frac{i \pi t}{2}}.
    $$
    """

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (
                0.0,
                np.array(
                    [[0.5, 0, 0, -0.5], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [-0.5, 0, 0, 0.5]]
                ),
            ),
            (
                1.0,
                np.array(
                    [[0.5, 0, 0, 0.5], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0.5, 0, 0, 0.5]]
                ),
            ),
        ]

    def _eigen_shifts(self):
        return [0, 1]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _decompose_into_clifford_with_qubits_(self, qubits):
        from cirq.ops.clifford_gate import SingleQubitCliffordGate
        from cirq.ops.pauli_interaction_gate import PauliInteractionGate

        if self.exponent % 2 == 0:
            return []
        if self.exponent % 2 == 0.5:
            return [
                PauliInteractionGate(pauli_gates.Y, False, pauli_gates.Y, False).on(*qubits),
                SingleQubitCliffordGate.Y_sqrt.on_each(*qubits),
            ]
        if self.exponent % 2 == 1:
            return [SingleQubitCliffordGate.Y.on_each(*qubits)]
        if self.exponent % 2 == 1.5:
            return [
                PauliInteractionGate(pauli_gates.Y, False, pauli_gates.Y, False).on(*qubits),
                SingleQubitCliffordGate.Y_nsqrt.on_each(*qubits),
            ]
        return NotImplemented

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> 'cirq.OP_TREE':
        yield common_gates.XPowGate(exponent=0.5).on_each(*qubits)
        yield ZZPowGate(exponent=self.exponent, global_shift=self.global_shift)(*qubits)
        yield common_gates.XPowGate(exponent=-0.5).on_each(*qubits)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('YY', 'YY'), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YY'
        return f'YY**{self._exponent!r}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YY'
            return f'(cirq.YY**{proper_repr(self._exponent)})'
        return (
            f'cirq.YYPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


@value.value_equality
class ZZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""The Z-parity gate, possibly raised to a power.

    The ZZ**t gate implements the following unitary:

    $$
    (Z \otimes Z)^t = \begin{bmatrix}
                      1 & & & \\
                      & e^{i \pi t} & & \\
                      & & e^{i \pi t} & \\
                      & & & 1
                      \end{bmatrix}
    $$
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits):
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[0])
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[1])
        yield common_gates.CZPowGate(
            exponent=-2 * self.exponent, global_shift=-self.global_shift / 2
        )(qubits[0], qubits[1])

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0, 0, 1])), (1, np.diag([0, 1, 1, 0]))]

    def _eigen_shifts(self):
        return [0, 1]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('ZZ', 'ZZ'), exponent=self._diagram_exponent(args)
        )

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if protocols.is_parameterized(self):
            return None

        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        if global_phase != 1:
            args.target_tensor *= global_phase

        relative_phase = 1j ** (2 * self.exponent)
        zo = args.subspace_index(0b01)
        oz = args.subspace_index(0b10)
        args.target_tensor[oz] *= relative_phase
        args.target_tensor[zo] *= relative_phase

        return args.target_tensor

    def _phase_by_(self, phase_turns: float, qubit_index: int) -> "ZZPowGate":
        return self

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZZ'
        return f'ZZ**{self._exponent}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZZ'
            return f'(cirq.ZZ**{proper_repr(self._exponent)})'
        return (
            f'cirq.ZZPowGate(exponent={proper_repr(self._exponent)}, '
            f'global_shift={self._global_shift!r})'
        )


class MSGate(XXPowGate):
    """The Mølmer–Sørensen gate, a native two-qubit operation in ion traps.

    A rotation around the XX axis in the two-qubit bloch sphere.

    The gate implements the following unitary:

        exp(-i t XX) = [ cos(t)   0        0       -isin(t)]
                       [ 0        cos(t)  -isin(t)  0      ]
                       [ 0       -isin(t)  cos(t)   0      ]
                       [-isin(t)  0        0        cos(t) ]
    """

    def __init__(self, *, rads: float):  # Forces keyword args.
        XXPowGate.__init__(self, exponent=rads * 2 / np.pi, global_shift=-0.5)
        self.rads = rads

    def _with_exponent(self, exponent: value.TParamVal) -> Self:
        return type(self)(rads=exponent * np.pi / 2)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args, order=4)
        symbol = f'MS({angle_str})'
        return protocols.CircuitDiagramInfo(wire_symbols=(symbol, symbol))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'MS(π/2)'
        return f'MS({self._exponent!r}π/2)'

    def __repr__(self) -> str:
        if self._exponent == 1:
            return 'cirq.ms(np.pi/2)'
        return f'cirq.ms({self._exponent!r}*np.pi/2)'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ["rads"])

    @classmethod
    def _from_json_dict_(cls, rads: float, **kwargs: Any) -> 'MSGate':
        return cls(rads=rads)


def ms(rads: float) -> MSGate:
    """A helper to construct the `cirq.MSGate` for the given angle specified in radians.

    Args:
        rads: The rotation angle in radians.

    Returns:
        Mølmer–Sørensen gate rotating by the desired amount.
    """
    return MSGate(rads=rads)


XX = XXPowGate()
document(
    XX,
    r"""The tensor product of two X gates.

    Useful for creating `cirq.XXPowGate`s via `cirq.XX**t`.

    This is the `exponent=1` instance of `cirq.XXPowGate`.
    """,
)
YY = YYPowGate()
document(
    YY,
    r"""The tensor product of two Y gates.

    Useful for creating `cirq.YYPowGate`s via `cirq.YY**t`.

    This is the `exponent=1` instance of `cirq.YYPowGate`.
    """,
)
ZZ = ZZPowGate()
document(
    ZZ,
    r"""The tensor product of two Z gates.

    Useful for creating `cirq.ZZPowGate`s via `cirq.ZZ**t`.

    This is the `exponent=1` instance of `cirq.ZZPowGate`.
    """,
)
