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

from typing import Mapping, Optional, Tuple, Union

import numpy as np

from cirq import protocols, value
from cirq.ops import gate_operation, raw_types


class LinearCombinationOfGates(value.LinearDict[raw_types.Gate]):
    """Represents linear operator defined by a linear combination of gates.

    Suppose G1, G2, ..., Gn are gates and b1, b2, ..., bn are complex
    numbers. Then

        LinearCombinationOfGates({G1: b1, G2: b2, ..., Gn: bn})

    represents the linear operator

        A = b1 G1 + b2 G2 + ... + bn Gn

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


class LinearCombinationOfGateOperations(
        value.LinearDict[gate_operation.GateOperation]):
    """Represents operator defined by linear combination of gate operations.

    If G1, ..., Gn are gate operations, {q1_1, ..., q1_k1}, {q2_1, ..., q2_k2},
    ..., {qn_1, ..., qn_kn} are (not necessarily disjoint) sets of qubits and
    b1, b2, ..., bn are complex numbers, then

        LinearCombinationOfGateOperations({
            G1(q1_1, ..., q1_k1): b1,
            G2(q2_1, ..., q2_k2): b2,
            ...,
            Gn(qn_1, ..., qn_kn): bn})

    represents the linear operator

        A = b1 G1(q1_1, ..., q1_k1) +
          + b2 G2(q2_1, ..., q2_k2) +
          + ... +
          + bn Gn(qn_1, ..., qn_kn)

    where in each term qubits not explicitly listed are assumed to be acted on
    by the identity operator. Note that A may not be unitary or even normal.

    Rather than creating LinearCombinationOfGateOperations instance explicitly,
    one may use overloaded arithmetic operators. For example,

        cirq.LinearCombinationOfGateOperations({cirq.X(q1): 2, cirq.Z(q2): -2})

    is equivalent to

        2 * cirq.X(q1) - 2 * cirq.Z(q2)

    TODO: Overloaded arithmetic operators are not implemented yet.
    """

    def __init__(self,
                 terms: Mapping[gate_operation.GateOperation, value.Scalar]
                ) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gate operations to coefficients in the linear
                combination being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def _is_compatible(self, gate: gate_operation.GateOperation) -> bool:
        return isinstance(gate, gate_operation.GateOperation)

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """Returns qubits acted on self."""
        if not self:
            return ()
        qubit_sets = [set(op.qubits) for op in self.keys()]
        all_qubits = set.union(*qubit_sets)
        return tuple(sorted(all_qubits))

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying operations.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """

        def extend(u: np.ndarray, qubits: Tuple[raw_types.Qid, ...],
                   all_qubits: Tuple[raw_types.Qid, ...]) -> np.ndarray:
            """Extends unitary on qubits to unitary on all_qubits."""
            assert qubits
            assert set(qubits).issubset(all_qubits)
            assert u.shape == (2**len(qubits), 2**len(qubits))

            if qubits[0] != all_qubits[0]:
                return np.kron(np.eye(2), extend(u, qubits, all_qubits[1:]))

            if len(qubits) == 1:
                return np.kron(u, np.eye(2**(len(all_qubits) - 1)))

            ab, cd = np.split(u, 2, axis=0)
            a, b = np.split(ab, 2, axis=1)
            c, d = np.split(cd, 2, axis=1)
            a = extend(a, qubits[1:], all_qubits[1:])
            b = extend(b, qubits[1:], all_qubits[1:])
            c = extend(c, qubits[1:], all_qubits[1:])
            d = extend(d, qubits[1:], all_qubits[1:])
            return np.block([[a, b], [c, d]])

        num_dim = 2**len(self.qubits)
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for op, coefficient in self.items():
            u = protocols.unitary(op)
            extended_u = extend(u, op.qubits, self.qubits)
            result += extended_u * coefficient
        return result

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        """Computes Pauli expansion of self from Pauli expansions of terms."""

        def extend_term(pauli_names: str, qubits: Tuple[raw_types.Qid, ...],
                        all_qubits: Tuple[raw_types.Qid, ...]) -> str:
            """Extends Pauli product on qubits to product on all_qubits."""
            assert len(pauli_names) == len(qubits)
            qubit_to_pauli_name = dict(zip(qubits, pauli_names))
            return ''.join(qubit_to_pauli_name.get(q, 'I') for q in all_qubits)

        def extend(expansion: value.LinearDict[str],
                   qubits: Tuple[raw_types.Qid, ...],
                   all_qubits: Tuple[raw_types.Qid, ...]
                  ) -> value.LinearDict[str]:
            """Extends Pauli expansion on qubits to expansion on all_qubits."""
            return value.LinearDict({
                extend_term(p, qubits, all_qubits): c
                for p, c in expansion.items()
            })

        result = value.LinearDict({})  # type: value.LinearDict[str]
        for op, coefficient in self.items():
            expansion = protocols.pauli_expansion(op)
            extended_expansion = extend(expansion, op.qubits, self.qubits)
            result += extended_expansion * coefficient
        return result
