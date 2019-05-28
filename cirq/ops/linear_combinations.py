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

from typing import Mapping, Optional, Union

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types


class LinearCombinationOfGates(value.LinearDict[raw_types.Gate]):
    """Represents linear operator defined by a linear combination of gates.

    Suppose G1, G2, ..., Gn are gates and b1, b2, ..., bn are complex
    numbers. Then

        LinearCombinationOfGates({G1: b1, G2: b2, ..., Gn: bn})

    represents the linear operator

        A = b1 G1 + b2 G2 + ... + bn * Gn

    Note that A may not be unitary or even normal.

    Rather than creating LinearCombinationOfGates instance explicitly, one may
    use overloaded arithmetic operators. For example,

        cirq.LinearCombinationOfGates({cirq.X: 2, cirq.Z: -2})

    is equivalent to

        2 * cirq.X - 2 * cirq.Z
    """
    def __init__(self, terms: Mapping[raw_types.Gate, value.Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gates to coefficients in the linear combination
                being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def num_qubits(self) -> Optional[int]:
        """Returns number of qubits in the domain if known, None if unknown."""
        if not self:
            return None
        any_gate = next(iter(self))
        return any_gate.num_qubits()

    def _is_compatible(self, gate: raw_types.Gate) -> bool:
        return (self.num_qubits() is None or
                self.num_qubits() == gate.num_qubits())

    def __add__(self,
                other: Union[raw_types.Gate, 'LinearCombinationOfGates']
                ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__add__(other)

    def __iadd__(self,
                 other: Union[raw_types.Gate, 'LinearCombinationOfGates']
                 ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__iadd__(other)

    def __sub__(self,
                other: Union[raw_types.Gate, 'LinearCombinationOfGates']
                ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__sub__(other)

    def __isub__(self,
                 other: Union[raw_types.Gate, 'LinearCombinationOfGates']
                 ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__isub__(other)

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying gates.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        num_qubits = self.num_qubits()
        if num_qubits is None:
            raise ValueError('Unknown number of qubits')
        num_dim = 2 ** num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for gate, coefficient in self.items():
            result += protocols.unitary(gate) * coefficient
        return result

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        result = value.LinearDict({})  # type: value.LinearDict[str]
        for gate, coefficient in self.items():
            result += protocols.pauli_expansion(gate) * coefficient
        return result
