# Copyright 2020 The Cirq Developers
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
from typing import (Any, Iterator, Tuple, Union, TYPE_CHECKING)

import numpy as np
import sympy

from cirq import linalg, protocols, value
from cirq.ops import linear_combinations, pauli_string_phasor

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliSumExponential:
    """Represents operator defined by exponential of a PauliSum.

    Given a hermitian/anti-hermitian PauliSum PS_1 + PS_2 + ... + PS_N, this
    class returns an operation which is equivalent to
    exp(j * exponent * (PS_1 + PS_2 + ... + PS_N)).

    This class currently supports only exponential of commuting Pauli Terms.
    """

    def __init__(self,
                 pauli_sum_like: 'cirq.PauliSumLike',
                 exponent: Union[int, float, sympy.Basic] = 1,
                 *,
                 is_anti_hermitian=False):
        pauli_sum = linear_combinations.PauliSum.wrap(pauli_sum_like)
        if not PauliSumExponential._all_pauli_strings_commute_(pauli_sum):
            raise ValueError(
                "PauliSumExponential defined only for commuting pauli sums.")
        self._exponent = exponent
        self._is_anti_hermitian = is_anti_hermitian
        if is_anti_hermitian:
            self._multiplier = -1j
        else:
            self._multiplier = 1
        for pauli_string in pauli_sum:
            normalised_coeff = complex(pauli_string.coefficient *
                                       self._multiplier)
            if normalised_coeff.imag != 0:
                raise ValueError(
                    pauli_string,
                    "PauliSum should be either hermitian or anti-hermitian.")
        self._pauli_sum = pauli_sum

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._pauli_sum.qubits

    def _value_equality_values_(self) -> Any:
        return (self._pauli_sum, self._exponent)

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum.with_qubits(*new_qubits),
                                   self._exponent,
                                   is_anti_hermitian=self._is_anti_hermitian)

    def _resolve_parameters_(self,
                             param_resolver: 'cirq.ParamResolverOrSimilarType'
                            ) -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum,
                                   exponent=protocols.resolve_parameters(
                                       self._exponent, param_resolver),
                                   is_anti_hermitian=self._is_anti_hermitian)

    @staticmethod
    def _all_pauli_strings_commute_(pauli_sum: 'cirq.PauliSum') -> bool:
        for x in pauli_sum:
            for y in pauli_sum:
                if not protocols.commutes(x, y):
                    return False
        return True

    def __iter__(self) -> Iterator['cirq.PauliStringPhasor']:
        for pauli_string in self._pauli_sum:
            theta = pauli_string.coefficient * self._multiplier
            theta *= self._exponent / np.pi
            if isinstance(theta, complex):
                theta = theta.real
            yield pauli_string_phasor.PauliStringPhasor(
                pauli_string.with_coefficient(1.0),
                exponent_neg=-theta,
                exponent_pos=theta)

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self from underlying Pauli sum exponentials.

        Raises:
            ValueError: if exponent is parameterized.
        """
        if protocols.is_parameterized(self._exponent):
            raise ValueError("Exponent should not parameterized.")
        ret = 1
        for pauli_string_exp in self:
            ret = np.kron(ret, protocols.unitary(pauli_string_exp))
        return ret

    def _has_unitary_(self) -> bool:
        return linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray:
        return self.matrix()

    def __pow__(self, exponent: int) -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum,
                                   self._exponent * exponent,
                                   is_anti_hermitian=self._is_anti_hermitian)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'cirq.{class_name}({self._pauli_sum!r}, {self._exponent!r}, '\
                'is_anti_hermitian={self._is_anti_hermitian!r})'

    def __str__(self) -> str:
        coeff = '1.0' if self._multiplier != 1 else 'j'
        return f'exp({(coeff)!s} * {self._exponent!s} * ({self._pauli_sum!s}))'
