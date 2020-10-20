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
from typing import (Any, Iterator, Tuple, TYPE_CHECKING)

import numpy as np
import scipy

from cirq import linalg, protocols, value
from cirq.ops import linear_combinations, pauli_string_phasor

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class PauliSumExponential:
    """Represents operator defined by exponential of a PauliSum.

    Given a PauliSum PS_1 + PS_2 + ... + PS_N, this class returns an operation
    which is equivalent to exp(i * exponent * (PS_1 + PS_2 + ... + PS_N)).

    This class currently supports only exponential of commuting Pauli Terms.
    """

    def __init__(self,
                 pauli_sum_like: 'cirq.PauliSumLike',
                 exponent: 'cirq.TParamVal' = 1):
        pauli_sum = linear_combinations.PauliSum.wrap(pauli_sum_like)
        if not PauliSumExponential._all_pauli_strings_commute_(pauli_sum):
            raise ValueError(
                "PauliSumExponential defined only for commuting pauli sums.")
        self._pauli_sum = pauli_sum
        self._exponent = exponent

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._pauli_sum.qubits

    @property
    def exponent(self) -> 'cirq.TParamVal':
        return self._exponent

    @property
    def pauli_sum(self) -> 'cirq.PauliSum':
        return self._pauli_sum

    def _value_equality_values_(self) -> Any:
        return (self._pauli_sum, self._exponent)

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum.with_qubits(*new_qubits),
                                   self._exponent)

    def copy(self) -> 'PauliSumExponential':
        factory = type(self)
        return factory(self._pauli_sum.copy(), self._exponent)

    def _resolve_parameters_(
        self, param_resolver: 'cirq.ParamResolverOrSimilarType'
    ) -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum,
                                   exponent=param_resolver.value_of(
                                       self._exponent))

    @staticmethod
    def _all_pauli_strings_commute_(pauli_sum: 'cirq.PauliSum') -> bool:
        for x in pauli_sum:
            for y in pauli_sum:
                if not protocols.commutes(x, y):
                    return False
        return True

    def __iter__(self) -> Iterator['cirq.PauliStringPhasor']:
        for pauli_string in self._pauli_sum:
            sign = np.sign(pauli_string.coefficient).real
            theta = self._exponent * sign / np.pi
            pauli_string.with_coefficient(1.0)
            yield pauli_string_phasor.PauliStringPhasor(
                pauli_string.with_coefficient(1.0),
                exponent_neg=-theta,
                exponent_pos=theta)

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self from underlying Pauli operations.

        Raises:
            ValueError: if exponent is parameterized.
        """
        if protocols.is_parameterized(self._exponent):
            raise ValueError("Exponent should not parameterized.")
        return scipy.linalg.expm(1j * self._exponent * self._pauli_sum.matrix())

    def _has_unitary_(self) -> bool:
        return linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def __pow__(self, exponent: int) -> 'PauliSumExponential':
        return PauliSumExponential(self._pauli_sum, self._exponent * exponent)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'cirq.{class_name}({self._pauli_sum!r}, {self._exponent!r})'

    def __str__(self) -> str:
        return f'exp(j * {self._exponent!s} * ({self._pauli_sum!s}))'
