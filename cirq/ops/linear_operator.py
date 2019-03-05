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

from typing import Dict, Optional, Tuple, Union
import abc

import numpy as np
import scipy.linalg

from cirq import protocols
from cirq.linalg import operator_spaces


class AbstractLinearOperator(metaclass=abc.ABCMeta):
    """General linear operator on a finite-dimensional vector space.

    AbstractLinearOperators encompass both quantum gates and other (not
    necessarily unitary) linear operators. Subclasses choose one or more
    internal representations by overriding some of the abstract methods
    below. This class in turn equips then with basic linear algebraic
    operations.
    """
    def num_qubits(self):
        return super().num_qubits()

    @abc.abstractmethod
    def _matrix_(self) -> Optional[np.ndarray]:
        """Returns the matrix representation of self."""
        pass

    @abc.abstractmethod
    def _pauli_expansion_(self) -> Dict[str, complex]:
        """Returns Pauli basis expansion of self.

        This representation is currently only used for single-qubit operators.
        """
        pass

    def matrix(self) -> Optional[np.ndarray]:
        """Returns or computes the matrix corresponding to self."""
        matrix = self._matrix_()
        if matrix is not None:
            return matrix

        pauli_expansion = self._pauli_expansion_()
        if pauli_expansion is not NotImplemented:
            return operator_spaces.matrix_from_basis_coefficients(
                pauli_expansion, operator_spaces.PAULI_BASIS)

        return None

    def __add__(
            self,
            other: 'AbstractLinearOperator'
    ) -> 'AbstractLinearOperator':
        """Computes vector addition in the space of linear operators."""
        pauli_expansion = None
        pauli_expansion_a = self._pauli_expansion_()
        pauli_expansion_b = other._pauli_expansion_()
        if (pauli_expansion_a is not NotImplemented and
                pauli_expansion_b is not NotImplemented):
            pauli_expansion = pauli_expansion_a.copy()
            for name, coefficient in pauli_expansion_b.items():
                if name in pauli_expansion:
                    pauli_expansion[name] += coefficient
                else:
                    pauli_expansion[name] = coefficient

        matrix = None
        matrix_a = self._matrix_()
        matrix_b = other._matrix_()
        if matrix_a is not None and matrix_b is not None:
            matrix = matrix_a + matrix_b

        return LinearOperator(matrix, pauli_expansion)

    def __sub__(
            self,
            other: 'AbstractLinearOperator'
    ) -> 'AbstractLinearOperator':
        """Computes difference between self and other."""
        return self.__add__(-other)

    def __neg__(self) -> 'AbstractLinearOperator':
        """Computes additive inverse of self."""
        return self.__mul__(-1)

    def __mul__(
            self,
            other: Union[complex, float, int]
    ) -> 'AbstractLinearOperator':
        """Computes scalar multiplication in the space of linear operators."""
        pauli_expansion = self._pauli_expansion_()
        if pauli_expansion is not NotImplemented:
            pauli_expansion = {
                    name: coefficient * other
                    for name, coefficient in pauli_expansion.items()
            }

        matrix = self._matrix_()
        if matrix is not None:
            matrix = matrix * other

        return LinearOperator(matrix, pauli_expansion)

    def __rmul__(
            self,
            other: Union[complex, float, int]
    ) -> 'AbstractLinearOperator':
        """Computes scalar multiplication in the space of linear operators."""
        return self.__mul__(other)

    def __truediv__(
            self,
            other: Union[complex, float, int]
    ) -> 'AbstractLinearOperator':
        return self.__mul__(1 / other)

    def __pow__(self, exponent: int) -> 'AbstractLinearOperator':
        """Computes integer power of a linear operator."""
        pauli_expansion = protocols.pauli_expansion(self, default=None)
        if pauli_expansion is None:
            return NotImplemented

        if self.num_qubits() != 1:
            return NotImplemented

        pauli_expansion = operator_spaces.operator_power(
            pauli_expansion, exponent)

        matrix = self._matrix_()
        if matrix is not None:
            matrix = operator_spaces.matrix_from_basis_coefficients(
                pauli_expansion, operator_spaces.PAULI_BASIS)

        return LinearOperator(matrix, pauli_expansion)

    def __rpow__(
            self,
            other: Union[complex, float, int]
    ) -> 'AbstractLinearOperator':
        """Computes operator power of a number.

        If self is skew hermitian, then the result is unitary.
        If self is hermitian, then the result is positive.
        """
        basis = operator_spaces.kron_bases(operator_spaces.PAULI_BASIS,
                                           repeat=self.num_qubits())

        matrix = self._matrix_()
        if matrix is None:
            matrix = operator_spaces.matrix_from_basis_coefficients(
                    self._pauli_expansion_(), basis)
        exponent = matrix * np.log(other)
        matrix = scipy.linalg.expm(exponent)

        pauli_expansion = self._pauli_expansion_()
        if pauli_expansion is not None:
            pauli_expansion = operator_spaces.expand_matrix_in_orthogonal_basis(
                matrix, basis)

        return LinearOperator(matrix, pauli_expansion)

    def exp(self) -> 'AbstractLinearOperator':
        """Computes operator power of number e."""
        return self.__rpow__(np.e)

    def polar_decomposition(self) -> Tuple['AbstractLinearOperator',
                                           'AbstractLinearOperator']:
        """Computes (right) polar decomposition of self.

        Returns a pair of operators (U, P) with U unitary and P positive
        (and hence hermitian) such that

            M = UP

        where M is self.
        """
        basis = operator_spaces.kron_bases(operator_spaces.PAULI_BASIS,
                                           repeat=self.num_qubits())

        matrix = self._matrix_()
        if matrix is None:
            pauli_expansion = self._pauli_expansion_()
            matrix = operator_spaces.matrix_from_basis_coefficients(
                pauli_expansion, operator_spaces.PAULI_BASIS)
        u_matrix, p_matrix = scipy.linalg.polar(matrix)

        u_pauli_expansion, p_pauli_expansion = None, None
        if self._pauli_expansion_() is not NotImplemented:
            u_pauli_expansion = (
                    operator_spaces.expand_matrix_in_orthogonal_basis(
                        u_matrix, basis))
            p_pauli_expansion = (
                    operator_spaces.expand_matrix_in_orthogonal_basis(
                        p_matrix, basis))

        return (LinearOperator(u_matrix, u_pauli_expansion),
                LinearOperator(p_matrix, p_pauli_expansion))

    def unitary_factor(self) -> 'AbstractLinearOperator':
        """Computes (right) unitary factor of self."""
        u, _ = self.polar_decomposition()
        return u


class UnitaryMixin(AbstractLinearOperator):
    """Mixin which equips a quantum gate with linear algebraic operations.

    To enable linear algebraic operations involving a gate, the gate must
    override one or both of _pauli_expansion_(), _unitary_().
    """
    def _matrix_(self) -> Optional[np.ndarray]:
        """Returns the matrix of the unitary operator."""
        return protocols.unitary(self, None)

    def _pauli_expansion_(self) -> Dict[str, complex]:
        """Returns the expansion of the unitary operator in the Pauli basis.

        Gates which know the expansion, should override.
        """
        return NotImplemented


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n-1) == 0


class LinearOperator(AbstractLinearOperator):
    """General linear operator on a finite-dimensional vector space.

    LinearOperator is never a gate. Instances of LinearOperator are used to
    represent intermediate values in manipulations of quantum gates that go
    outside of U(n). It is possible to return to U(n) by performing polar
    decomposition, see above.

    Alternatively, use cirq.make_gate() or cirq.forge_gate() to perform polar
    decomposition implicitly and obtain a quantum gate instance.

    TODO(viathor): Use LinearOperators to describe measurement and enable
                   efficient computation of averages.
    """
    def __init__(self,
                 matrix: Optional[np.ndarray] = None,
                 pauli_expansion: Optional[Dict[str, complex]] = None) -> None:
        self._matrix = matrix
        self._pauli_expansion = pauli_expansion

        if matrix is not None:
            self._validate_matrix(matrix)
            self._num_qubits = matrix.shape[0].bit_length() - 1
        elif pauli_expansion is not None:
            self._num_qubits = len(next(iter(pauli_expansion.keys())))
        else:
            raise ValueError('Either matrix or Pauli basis expansion must be '
                             'provided to initialize LinearOperator.')

    def num_qubits(self) -> int:
        return self._num_qubits

    @staticmethod
    def _validate_matrix(matrix: np.ndarray) -> None:
        if len(matrix.shape) != 2:
            raise ValueError(
                'Cannot make LinearOperator from {}-dimensional array'
                .format(len(matrix.shape)))
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                'Cannot make LinearOperator from matrix of shape {}'
                .format(matrix.shape))
        if not _is_power_of_two(matrix.shape[0]):
            raise ValueError(
                'Cannot make LinearOperator from matrix os size {}x{}'
                .format(*matrix.shape))

    def _matrix_(self) -> Optional[np.ndarray]:
        return self._matrix

    def _pauli_expansion_(self) -> Dict[str, complex]:
        if self._pauli_expansion is None:
            return NotImplemented
        return self._pauli_expansion

    @staticmethod
    def _term_to_str(name: str, coefficient: complex) -> str:
        if abs(coefficient.real) < 1e-4 and abs(coefficient.imag) < 1e-4:
            return ''
        if abs(coefficient.real) < 1e-4:
            return '{:+.3f}i*{}'.format(coefficient.imag, name)
        if abs(coefficient.imag) < 1e-4:
            return '{:+.3f}*{}'.format(coefficient.real, name)
        return '+({:.3f}{:+.3f}i)*{}'.format(
                coefficient.real, coefficient.imag, name)

    def __str__(self):
        pauli_expansion = self._pauli_expansion_()
        if pauli_expansion is not NotImplemented:
            terms = [(name, pauli_expansion[name])
                     for name in sorted(pauli_expansion.keys())]
            s = ''.join(self._term_to_str(n, c) for n, c in terms)
            return s[1:] if s else '0'
        return str(self._matrix.round(3))

    def __repr__(self):
        if self._pauli_expansion is None:
            return 'cirq.LinearOperator(np.array({!r}))'.format(
                self._matrix.tolist())
        if self._matrix is None:
            return 'cirq.LinearOperator(pauli_expansion={!r})'.format(
                self._pauli_expansion)
        return (
            'cirq.LinearOperator(matrix=np.array({!r}), '
            'pauli_expansion={!r})'.format(
                self._matrix.tolist(), self._pauli_expansion))
