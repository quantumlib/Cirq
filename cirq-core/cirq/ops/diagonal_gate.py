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

"""Creates the gate instance for any number qubits diagonal gate.

The gate is used to create a (2^n)x(2^n) matrix with the diagonal elements
passed as a list.
"""

from typing import (
    AbstractSet,
    Any,
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

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import common_gates, raw_types, global_phase_op

if TYPE_CHECKING:
    import cirq


def _fast_walsh_hadamard_transform(a: Tuple[Any, ...]) -> np.ndarray:
    """Fast Walsh–Hadamard Transform of an array."""
    h = 1
    a_ = np.array(a)
    while h < len(a_):
        for i in range(0, len(a_), h * 2):
            for j in range(i, i + h):
                x = a_[j]
                y = a_[j + h]
                a_[j] = x + y
                a_[j + h] = x - y
        h *= 2
    return a_


def _gen_gray_code(n: int) -> Iterator[Tuple[int, int]]:
    """Generate the Gray Code from 0 to 2^n-1.

    Each iteration yields a two-tuple, `(gray_code, bit_flip)`. `gray_code` is the decimal
    representation of the gray code and `bit_flip` is the position of bits flipped for next
    gray code.
    """
    gray_code = 0
    for i in range(1, 2**n):
        next_gray = i ^ (i >> 1)
        bit_flip = int(np.log2(gray_code ^ next_gray))
        yield gray_code, bit_flip
        gray_code = next_gray
    yield gray_code, int(np.log2(gray_code))


@value.value_equality()
class DiagonalGate(raw_types.Gate):
    r"""An n qubit gate which acts as phases on computational basis states.

    This gate's off-diagonal elements are zero and its on-diagonal elements are
    all phases.
    """

    def __init__(self, diag_angles_radians: Sequence['cirq.TParamVal']) -> None:
        r"""A n-qubit gate with only diagonal elements.

        This gate's off-diagonal elements are zero and its on-diagonal
        elements are all phases.

        Args:
            diag_angles_radians: The list of angles on the diagonal in radians.
                If these values are $(x_0, x_1, \ldots , x_N)$ then the unitary
                has diagonal values $(e^{i x_0}, e^{i x_1}, \ldots, e^{i x_N})$.
        """
        self._diag_angles_radians: Tuple['cirq.TParamVal', ...] = tuple(diag_angles_radians)

    @property
    def diag_angles_radians(self) -> Tuple['cirq.TParamVal', ...]:
        return self._diag_angles_radians

    def _num_qubits_(self):
        return int(np.log2(len(self._diag_angles_radians)))

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(angle) for angle in self._diag_angles_radians)

    def _parameter_names_(self) -> AbstractSet[str]:
        return {
            name for angle in self._diag_angles_radians for name in protocols.parameter_names(angle)
        }

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'DiagonalGate':
        return DiagonalGate(
            protocols.resolve_parameters(self._diag_angles_radians, resolver, recursive)
        )

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        return np.diag([np.exp(1j * angle) for angle in self._diag_angles_radians])

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if self._is_parameterized_():
            return NotImplemented
        for index, angle in enumerate(self._diag_angles_radians):
            subspace_index = args.subspace_index(big_endian_bits_int=index)
            args.target_tensor[subspace_index] *= np.exp(1j * angle)
        return args.target_tensor

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        rounded_angles = np.array(self._diag_angles_radians)
        if args.precision is not None:
            rounded_angles = rounded_angles.round(args.precision)
        if len(rounded_angles) <= 4:
            rounded_angles_str = ', '.join(proper_repr(angle) for angle in rounded_angles)
            diag_str = f'diag({rounded_angles_str})'
        else:
            diag_str = ', '.join(proper_repr(angle) for angle in rounded_angles[:2])
            diag_str += ', ..., '
            diag_str += ', '.join(proper_repr(angle) for angle in rounded_angles[-2:])
            diag_str = f'diag({diag_str})'
        return protocols.CircuitDiagramInfo(
            [diag_str] + [f"#{i}" for i in range(2, self._num_qubits_() + 1)]
        )

    def __pow__(self, exponent: Any) -> 'DiagonalGate':
        if not isinstance(exponent, (int, float, sympy.Basic)):
            return NotImplemented
        angles = []
        for angle in self._diag_angles_radians:
            mul_angle = protocols.mul(angle, exponent, NotImplemented)
            angles.append(mul_angle)
        return DiagonalGate(angles)

    def _value_equality_values_(self) -> Any:
        return tuple(self._diag_angles_radians)

    def _decompose_for_basis(
        self, index: int, bit_flip: int, theta: 'cirq.TParamVal', qubits: Sequence['cirq.Qid']
    ) -> Iterator[Union['cirq.ZPowGate', 'cirq.CXPowGate']]:
        if index == 0:
            return []
        largest_digit = self._num_qubits_() - (len(bin(index)) - 2)
        yield common_gates.rz(2 * theta)(qubits[largest_digit])
        _flip_bit = self._num_qubits_() - bit_flip - 1
        if _flip_bit < largest_digit:
            yield common_gates.CNOT(qubits[largest_digit], qubits[_flip_bit])
        elif _flip_bit > largest_digit:
            yield common_gates.CNOT(qubits[_flip_bit], qubits[largest_digit])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Decompose the n-qubit diagonal gates into CNOT and Rz gates.

        A 3 qubits decomposition looks like
        0: ───────────────────────────────────X───Rz(6)───X───Rz(7)───X───Rz(5)───X───Rz(4)───
                                              │           │           │           │
        1: ───────────X───Rz(3)───X───Rz(2)───@───────────┼───────────@───────────┼───────────
                      │           │                       │                       │
        2: ───Rz(1)───@───────────@───────────────────────@───────────────────────@───────────

        where the angles in Rz gates are corresponding to the fast-walsh-Hadamard transform
        of diagonal_angles in the Gray Code order.

        For n qubits decomposition looks similar but with 2^n-1 Rz gates and 2^n-2 CNOT gates.

        The algorithm is implemented according to the paper:
            Welch, Jonathan, et al. "Efficient quantum circuits for diagonal unitaries without
            ancillas." New Journal of Physics 16.3 (2014): 033040.
            https://iopscience.iop.org/article/10.1088/1367-2630/16/3/033040/meta
        """
        n = self._num_qubits_()
        hat_angles = _fast_walsh_hadamard_transform(self._diag_angles_radians) / (2**n)

        # There is one global phase shift between unitary matrix of the diagonal gate and the
        # decomposed gates. On its own it is not physically observable. However, if using this
        # diagonal gate for sub-system like controlled gate, it is no longer equivalent. Hence,
        # we add global phase.
        decomposed_circ: List[Any] = [
            global_phase_op.global_phase_operation(1j ** (2 * hat_angles[0] / np.pi))
        ]
        for i, bit_flip in _gen_gray_code(n):
            decomposed_circ.extend(self._decompose_for_basis(i, bit_flip, -hat_angles[i], qubits))
        return decomposed_circ

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=["diag_angles_radians"])

    def __repr__(self) -> str:
        angles = ','.join(proper_repr(angle) for angle in self._diag_angles_radians)
        return f'cirq.DiagonalGate([{angles}])'
