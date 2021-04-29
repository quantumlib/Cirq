# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");l
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

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from cirq import protocols
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import gate_features, eigen_gate, common_gates, pauli_gates

if TYPE_CHECKING:
    import cirq


class XXPowGate(
    eigen_gate.EigenGate, gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate
):
    r"""The X-parity gate, possibly raised to a power.

    The XX**t gate implements the following unitary:

        $$
        (X \otimes X)^t = \begin{bmatrix}
                          c & . & . & s \\
                          . & c & s & . \\
                          . & s & c & . \\
                          s & . & . & c
                          \end{bmatrix}
        $$

    where '.' means '0' and

        $$
        c = f \cos(\frac{\pi t}{2})
        $$

        $$
        s = -i f \sin(\frac{\pi t}{2})
        $$

        $$
        f = e^{\frac{i \pi t}{2}}.
        $$

    See also: `cirq.ion.ion_gates.MSGate` (the Mølmer–Sørensen gate), which is
    implemented via this class.
    """

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

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX', 'XX'), exponent=self._diagram_exponent(args)
        )

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('X {0}\nX {1}\n', qubits[0], qubits[1])
        return formatter.format(
            'RX({0}) {1}\nRX({2}) {3}\n',
            self._exponent * np.pi,
            qubits[0],
            self._exponent * np.pi,
            qubits[1],
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


class YYPowGate(
    eigen_gate.EigenGate, gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate
):
    r"""The Y-parity gate, possibly raised to a power.

    The YY**t gate implements the following unitary:

        $$
        (Y \otimes Y)^t = \begin{bmatrix}
                          c & . & . & -s \\
                          . & c & s & . \\
                          . & s & c & . \\
                          -s & . & . & c \\
                          \end{bmatrix}
        $$

    where '.' means '0' and

        $$
        c = f \cos(\frac{\pi t}{2})
        $$

        $$
        s = -i f \sin(\frac{\pi t}{2})
        $$

        $$
        f = e^{\frac{i \pi t}{2}}.
        $$
    """

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

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('YY', 'YY'), exponent=self._diagram_exponent(args)
        )

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('Y {0}\nY {1}\n', qubits[0], qubits[1])

        return formatter.format(
            'RY({0}) {1}\nRY({2}) {3}\n',
            self._exponent * np.pi,
            qubits[0],
            self._exponent * np.pi,
            qubits[1],
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


class ZZPowGate(
    eigen_gate.EigenGate, gate_features.TwoQubitGate, gate_features.InterchangeableQubitsGate
):
    r"""The Z-parity gate, possibly raised to a power.

    The ZZ**t gate implements the following unitary:

        $$
        (Z \otimes Z)^t = \begin{bmatrix}
                          1 & . & . & . \\
                          . & w & . & . \\
                          . & . & w & . \\
                          . & . & . & 1
                          \end{bmatrix}
        $$

    where $w = e^{i \pi t}$ and '.' means '0'.
    """

    def _decompose_(self, qubits):
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[0])
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[1])
        yield common_gates.CZPowGate(
            exponent=-2 * self.exponent, global_shift=-self.global_shift / 2
        )(qubits[0], qubits[1])

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.diag([1, 0, 0, 1])),
            (1, np.diag([0, 1, 1, 0])),
        ]

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

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if self._exponent == 1:
            return formatter.format('Z {0}\nZ {1}\n', qubits[0], qubits[1])

        return formatter.format(
            'RZ({0}) {1}\nRZ({2}) {3}\n',
            self._exponent * np.pi,
            qubits[0],
            self._exponent * np.pi,
            qubits[1],
        )

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


XX = XXPowGate()
document(
    XX,
    """The tensor product of two X gates.

    The `exponent=1` instance of `cirq.XXPowGate`.
    """,
)
YY = YYPowGate()
document(
    YY,
    """The tensor product of two Y gates.

    The `exponent=1` instance of `cirq.YYPowGate`.
    """,
)
ZZ = ZZPowGate()
document(
    ZZ,
    """The tensor product of two Z gates.

    The `exponent=1` instance of `cirq.ZZPowGate`.
    """,
)
