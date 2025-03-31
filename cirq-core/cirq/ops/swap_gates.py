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
"""SWAP and ISWAP gates.

This module creates Gate instances for the following gates:
    SWAP: the swap gate.
    ISWAP: a swap gate with a phase on the swapped subspace.
    ISWAP_INV: the inverse of the ISWAP gate.
    SQRT_ISWAP: square root of the ISWAP gate.
    SQRT_ISWAP_INV: inverse square root of the ISWAP gate.

Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. SQRT_ISWAP_INV=cirq.ISWAP**-0.5). See the definition in
EigenGate.
"""

from typing import cast, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import sympy

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import common_gates, eigen_gate, gate_features

if TYPE_CHECKING:
    import cirq


class SwapPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""The SWAP gate, possibly raised to a power. Exchanges qubits.

    SwapPowGate()**t = SwapPowGate(exponent=t) and acts on two qubits in the
    computational basis as the matrix:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & g c & -i g s & 0 \\
        0 & -i g s & g c & 0 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
    $$

    where:

    $$
    c = \cos\left(\frac{\pi t}{2}\right)
    $$
    $$
    s = \sin\left(\frac{\pi t}{2}\right)
    $$
    $$
    g = e^{\frac{i \pi t}{2}}
    $$

    `cirq.SWAP`, the swap gate, is an instance of this gate at exponent=1.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits):
        """See base class."""
        a, b = qubits
        yield common_gates.CNOT(a, b)
        yield common_gates.CNotPowGate(exponent=self._exponent, global_shift=self.global_shift).on(
            b, a
        )
        yield common_gates.CNOT(a, b)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        # yapf: disable
        return [
            (0, np.array([[1, 0,   0,   0],
                          [0, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 0],
                          [0, 0,   0,   1]])),
            (1, np.array([[0,  0,    0,   0],
                          [0,  0.5, -0.5, 0],
                          [0, -0.5,  0.5, 0],
                          [0,  0,    0,   0]])),
        ]
        # yapf: enable

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented

        zo = args.subspace_index(0b01)
        oz = args.subspace_index(0b10)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        swap_phase = 1j**self._exponent
        c = -1j * swap_phase * np.sin(np.pi * self._exponent / 2) / 2
        return value.LinearDict(
            {
                'II': global_phase * (1 - c),
                'XX': global_phase * c,
                'YY': global_phase * c,
                'ZZ': global_phase * c,
            }
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(
                wire_symbols=('Swap', 'Swap'), exponent=self._diagram_exponent(args)
            )
        return protocols.CircuitDiagramInfo(
            wire_symbols=('×', '×'), exponent=self._diagram_exponent(args)
        )

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0', '3.0')
        return args.format('swap {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'SWAP'
        return f'SWAP**{self._exponent}'

    def __repr__(self) -> str:
        e = proper_repr(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.SWAP'
            return f'(cirq.SWAP**{e})'
        return f'cirq.SwapPowGate(exponent={e}, global_shift={self._global_shift!r})'


class ISwapPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    r"""Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

    When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by i. More
    generally, this gate's matrix is defined as follows:

        ISWAP**t ≡ exp(+i π t (X⊗X + Y⊗Y) / 4)

    which is given by the matrix:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & c & i s & 0 \\
        0 & i s & c & 0 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
    $$

    where

    $$
    c = \cos\left(\frac{\pi t}{2}\right)
    $$
    $$
    s = \sin\left(\frac{\pi t}{2}\right)
    $$

    `cirq.ISWAP`, the swap gate that applies i to the |01⟩ and |10⟩ states,
    is an instance of this gate at exponent=1.

    References:
        "What is the matrix of the iSwap gate?"
        https://quantumcomputing.stackexchange.com/questions/2594/
    """

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        # yapf: disable
        return [
            (0, np.diag([1, 0, 0, 1])),
            (+0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0, 0, 0]])),
            (-0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, -0.5, 0],
                             [0, -0.5, 0.5, 0],
                             [0, 0, 0, 0]])),
        ]
        # yapf: enable

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def _decompose_(self, qubits):
        a, b = qubits

        yield common_gates.CNOT(a, b)
        yield common_gates.H(a)
        yield common_gates.CNOT(b, a)
        yield common_gates.ZPowGate(exponent=self._exponent / 2, global_shift=self.global_shift).on(
            a
        )
        yield common_gates.CNOT(b, a)
        yield common_gates.ZPowGate(
            exponent=-self._exponent / 2, global_shift=-self.global_shift
        ).on(a)
        yield common_gates.H(a)
        yield common_gates.CNOT(a, b)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented

        zo = args.subspace_index(0b01)
        oz = args.subspace_index(0b10)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        args.target_tensor[zo] *= 1j
        args.target_tensor[oz] *= 1j
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        angle = np.pi * self._exponent / 4
        c, s = np.cos(angle), np.sin(angle)
        return value.LinearDict(
            {
                'II': global_phase * c * c,
                'XX': global_phase * c * s * 1j,
                'YY': global_phase * s * c * 1j,
                'ZZ': global_phase * s * s,
            }
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('iSwap', 'iSwap'), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ISWAP'
        if self._exponent == -1:
            return 'ISWAP_INV'
        return f'ISWAP**{self._exponent}'

    def __repr__(self) -> str:
        e = proper_repr(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ISWAP'
            if self._exponent == -1:
                return 'cirq.ISWAP_INV'
            return f'(cirq.ISWAP**{e})'
        return f'cirq.ISwapPowGate(exponent={e}, global_shift={self._global_shift!r})'


def riswap(rads: value.TParamVal) -> ISwapPowGate:
    """Returns gate with matrix exp(+i angle_rads (X⊗X + Y⊗Y) / 2)."""
    pi = sympy.pi if protocols.is_parameterized(rads) else np.pi
    return cast(ISwapPowGate, ISwapPowGate() ** (2 * rads / pi))


SWAP = SwapPowGate()
document(
    SWAP,
    r"""The swap gate.

    This gate will swap two qubits (in any basis).

    The unitary matrix of this gate is:
    $$
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}
    $$
    """,
)

ISWAP = ISwapPowGate()
document(
    ISWAP,
    r"""The iswap gate.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & i & 0 \\
        0 & i & 0 & 0 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
    $$
    """,
)

ISWAP_INV = ISwapPowGate(exponent=-1)
document(
    ISWAP_INV,
    r"""The inverse of the iswap gate.

    The unitary matrix of this gate is:
    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & -i & 0 \\
        0 & -i & 0 & 0 \\
        0 & 0 & 0 & 1
    \end{bmatrix}
    $$
    """,
)

SQRT_ISWAP = ISwapPowGate(exponent=0.5)
document(
    SQRT_ISWAP,
    r"""The square root of iswap gate.

    The unitary matrix of this gate is:
    $$
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & \frac{1}{\sqrt{2}} & \frac{i}{\sqrt{2}} & 0 \\
            0 & \frac{i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}
    $$
    """,
)

SQRT_ISWAP_INV = ISwapPowGate(exponent=-0.5)
document(
    SQRT_ISWAP_INV,
    r"""The inverse square root of iswap gate.

    The unitary matrix of this gate is:
    $$
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & \frac{1}{\sqrt{2}} & -\frac{i}{\sqrt{2}} & 0 \\
            0 & -\frac{i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}
    $$
    """,
)
