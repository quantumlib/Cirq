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
"""ISWAPPowGate conjugated by tensor product Rz(phi) and Rz(-phi)."""

from typing import AbstractSet, Any, cast, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy

import cirq
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import eigen_gate, swap_gates


@value.value_equality(manual_cls=True)
class PhasedISwapPowGate(eigen_gate.EigenGate):
    r"""Fractional ISWAP conjugated by Z rotations.

    PhasedISwapPowGate with phase_exponent p and exponent t is equivalent to
    the composition

        (Z^-p ⊗ Z^p) ISWAP^t (Z^p ⊗ Z^-p)

    and is given by the matrix:

    $$
    \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & c & i s f & 0 \\
        0 & i s f^* & c & 0 \\
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
    f = e^{2 \pi p i}
    $$
    """

    def __init__(
        self,
        *,
        phase_exponent: Union[float, sympy.Expr] = 0.25,
        exponent: Union[float, sympy.Expr] = 1.0,
        global_shift: float = 0.0,
    ):
        """Inits PhasedISwapPowGate.

        Args:
            phase_exponent: The exponent on the Z gates. We conjugate by
                the T gate by default.
            exponent: The exponent on the ISWAP gate, see EigenGate for
                details.
            global_shift: The global_shift on the ISWAP gate, see EigenGate for
                details.
        """
        self._phase_exponent = value.canonicalize_half_turns(phase_exponent)
        self._iswap = swap_gates.ISwapPowGate(exponent=exponent, global_shift=global_shift)
        super().__init__(exponent=exponent, global_shift=global_shift)

    @property
    def phase_exponent(self) -> Union[float, sympy.Expr]:
        return self._phase_exponent

    def _num_qubits_(self) -> int:
        return 2

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'phase_exponent': self._phase_exponent,
            'exponent': self._exponent,
            'global_shift': self._global_shift,
        }

    def _value_equality_values_cls_(self):
        return PhasedISwapPowGate

    def _value_equality_values_(self):
        return (self.phase_exponent, *self._iswap._value_equality_values_())

    def _value_equality_approximate_values_(self):
        return (self.phase_exponent, *self._iswap._value_equality_approximate_values_())

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._iswap) or protocols.is_parameterized(
            self._phase_exponent
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._iswap) | protocols.parameter_names(
            self._phase_exponent
        )

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PhasedISwapPowGate':
        return self.__class__(
            phase_exponent=protocols.resolve_parameters(self.phase_exponent, resolver, recursive),
            exponent=protocols.resolve_parameters(self.exponent, resolver, recursive),
        )

    def _with_exponent(self, exponent: value.type_alias.TParamVal) -> 'PhasedISwapPowGate':
        return PhasedISwapPowGate(
            phase_exponent=self.phase_exponent, exponent=exponent, global_shift=self.global_shift
        )

    def _eigen_shifts(self) -> List[float]:
        return [0.0, +0.5, -0.5]

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        phase = np.exp(1j * np.pi * self.phase_exponent)
        phase_matrix = np.diag([1, phase, phase.conjugate(), 1])
        inverse_phase_matrix = np.conjugate(phase_matrix)
        eigen_components: List[Tuple[float, np.ndarray]] = []
        for eigenvalue, projector in self._iswap._eigen_components():
            new_projector = phase_matrix @ projector @ inverse_phase_matrix
            eigen_components.append((eigenvalue, new_projector))
        return eigen_components

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if protocols.is_parameterized(self):
            return NotImplemented

        c = np.cos(np.pi * self._exponent / 2)
        s = np.sin(np.pi * self._exponent / 2)
        f = np.exp(2j * np.pi * self._phase_exponent)
        matrix = np.array([[c, 1j * s * f], [1j * s * f.conjugate(), c]])
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p

        zo = args.subspace_index(0b01)
        oz = args.subspace_index(0b10)
        linalg.apply_matrix_to_slices(
            args.target_tensor, matrix, [oz, zo], out=args.available_buffer
        )
        return args.available_buffer

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> Iterator['cirq.OP_TREE']:
        if len(qubits) != 2:
            raise ValueError(f'Expected two qubits, got {len(qubits)}')
        a, b = qubits

        yield cirq.Z(a) ** self.phase_exponent
        yield cirq.Z(b) ** -self.phase_exponent
        yield cirq.ISwapPowGate(exponent=self.exponent, global_shift=self.global_shift).on(a, b)
        yield cirq.Z(a) ** -self.phase_exponent
        yield cirq.Z(b) ** self.phase_exponent

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if self._is_parameterized_():
            return NotImplemented
        expansion = protocols.pauli_expansion(self._iswap)
        assert set(expansion.keys()).issubset({'II', 'XX', 'YY', 'ZZ'})
        assert np.isclose(cast(np.generic, expansion['XX']), cast(np.generic, expansion['YY']))

        v = (expansion['XX'] + expansion['YY']) / 2
        phase_angle = np.pi * self.phase_exponent
        c, s = np.cos(2 * phase_angle), np.sin(2 * phase_angle)

        return value.LinearDict(
            {
                'II': expansion['II'],
                'XX': c * v,
                'YY': c * v,
                'XY': s * v,
                'YX': -s * v,
                'ZZ': expansion['ZZ'],
            }
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        s = f'PhISwap({args.format_real(self._phase_exponent)})'
        return protocols.CircuitDiagramInfo(
            wire_symbols=(s, s), exponent=self._diagram_exponent(args)
        )

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self.exponent == 1:
                return 'PhasedISWAP'
            return f'PhasedISWAP**{self.exponent}'
        return f'PhasedISWAP(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        phase_exponent = proper_repr(self._phase_exponent)
        args = [f'phase_exponent={phase_exponent}']
        if self.exponent != 1:
            exponent = proper_repr(self.exponent)
            args.append(f'exponent={exponent}')
        if self._global_shift != 0:
            args.append(f'global_shift={self._global_shift}')
        arg_string = ', '.join(args)
        return f'cirq.PhasedISwapPowGate({arg_string})'


def givens(angle_rads: value.TParamVal) -> PhasedISwapPowGate:
    """Returns gate with matrix exp(-i angle_rads (Y⊗X - X⊗Y) / 2).

    In numerical linear algebra Givens rotation is any linear transformation
    with matrix equal to the identity except for a 2x2 orthogonal submatrix
    [[cos(a), -sin(a)], [sin(a), cos(a)]] which performs a 2D rotation on a
    subspace spanned by two basis vectors. In quantum computational chemistry
    the term is used to refer to the two-qubit gate defined as

        givens(a) ≡ exp(-i a (Y⊗X - X⊗Y) / 2)

    with the matrix

        [[1, 0, 0, 0],
         [0, c, -s, 0],
         [0, s, c, 0],
         [0, 0, 0, 1]]

    where

        c = cos(a),
        s = sin(a).

    The matrix is a Givens rotation in the numerical linear algebra sense
    acting on the subspace spanned by the |01⟩ and |10⟩ states.

    The gate is also equivalent to the ISWAP conjugated by T^-1 ⊗ T.


    Args:
        angle_rads: The rotation angle in radians.

    Returns:
        A phased iswap gate for the given rotation.
    """
    pi = sympy.pi if protocols.is_parameterized(angle_rads) else np.pi
    return cast(PhasedISwapPowGate, PhasedISwapPowGate() ** (2 * angle_rads / pi))
