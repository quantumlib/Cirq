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
from collections import defaultdict
from typing import Mapping, Optional, Tuple, Union, List, FrozenSet, DefaultDict

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString

UnitPauliStringT = FrozenSet[Tuple[raw_types.Qid, pauli_gates.Pauli]]
PauliSumLike = Union[int, float, complex, PauliString, 'PauliSum', pauli_string.
                     SingleQubitPauliStringGateOperation]


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


class LinearCombinationOfOperations(value.LinearDict[raw_types.Operation]):
    """Represents operator defined by linear combination of gate operations.

    If G1, ..., Gn are gate operations, {q1_1, ..., q1_k1}, {q2_1, ..., q2_k2},
    ..., {qn_1, ..., qn_kn} are (not necessarily disjoint) sets of qubits and
    b1, b2, ..., bn are complex numbers, then

        LinearCombinationOfOperations({
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
    """

    def __init__(self,
                 terms: Mapping[raw_types.Operation, value.Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gate operations to coefficients in the linear
                combination being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def _is_compatible(self, operation: raw_types.Operation) -> bool:
        return isinstance(operation, raw_types.Operation)

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
        num_qubits = len(self.qubits)
        num_dim = 2**num_qubits
        qubit_to_axis = {q: i for i, q in enumerate(self.qubits)}
        result = np.zeros((2,) * (2 * num_qubits), dtype=np.complex128)
        for op, coefficient in self.items():
            identity = np.eye(num_dim,
                              dtype=np.complex128).reshape(result.shape)
            workspace = np.empty_like(identity)
            axes = tuple(qubit_to_axis[q] for q in op.qubits)
            u = protocols.apply_unitary(
                op, protocols.ApplyUnitaryArgs(identity, workspace, axes))
            result += coefficient * u
        return result.reshape((num_dim, num_dim))

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


def _is_linear_dict_of_unit_pauli_string(
        linear_dict: value.LinearDict[UnitPauliStringT]) -> bool:
    if not isinstance(linear_dict, value.LinearDict):
        return False
    for k in linear_dict.keys():
        if not isinstance(k, frozenset):
            return False
        for qid, pauli in k:
            if not isinstance(qid, raw_types.Qid):
                return False
            if not isinstance(pauli, pauli_gates.Pauli):
                return False

    return True


def _pauli_string_from_unit(unit: UnitPauliStringT,
                            coefficient: Union[int, float, complex] = 1):
    return PauliString(dict(unit), coefficient=coefficient)


@value.value_equality(approximate=True)
class PauliSum:
    """Represents operator defined by linear combination of PauliStrings.

    Since PauliStrings store their own coefficients, this class
    does not implement the LinearDict interface. Instead, you can
    add and subtract terms and then iterate over the resulting
    (simplified) expression.

    Under the hood, this class is backed by a LinearDict with coefficient-less
    PauliStrings as keys. PauliStrings are reconstructed on-the-fly during
    iteration.
    """

    def __init__(
            self,
            linear_dict: Optional[value.LinearDict[UnitPauliStringT]] = None):
        if linear_dict is None:
            linear_dict = value.LinearDict()
        if not _is_linear_dict_of_unit_pauli_string(linear_dict):
            raise ValueError(
                "PauliSum constructor takes a LinearDict[UnitPauliStringT]. "
                "Consider using PauliSum.from_pauli_strings() or adding and "
                "subtracting PauliStrings")
        self._linear_dict = linear_dict

    def _value_equality_values_(self):
        return self._linear_dict

    @staticmethod
    def wrap(val: PauliSumLike) -> 'PauliSum':
        if isinstance(val, PauliSum):
            return val
        return PauliSum() + val

    @classmethod
    def from_pauli_strings(cls, terms: Union[PauliString, List[PauliString]]) \
            -> 'PauliSum':
        if isinstance(terms, PauliString):
            terms = [terms]
        termdict = defaultdict(
            lambda: 0)  # type: DefaultDict[UnitPauliStringT, value.Scalar]
        for pstring in terms:
            key = frozenset(pstring._qubit_pauli_map.items())
            termdict[key] += pstring.coefficient
        return cls(linear_dict=value.LinearDict(termdict))

    def copy(self) -> 'PauliSum':
        factory = type(self)
        return factory(self._linear_dict.copy())

    def __iter__(self):
        for vec, coeff in self._linear_dict.items():
            yield _pauli_string_from_unit(vec, coeff)

    def __len__(self) -> int:
        return len(self._linear_dict)

    def __iadd__(self, other):
        if isinstance(other, (float, int, complex)):
            other = PauliSum.from_pauli_strings(
                [PauliString(coefficient=other)])
        elif isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])

        if not isinstance(other, PauliSum):
            return NotImplemented

        self._linear_dict += other._linear_dict
        return self

    def __add__(self, other):
        if not isinstance(other, (float, int, complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        if isinstance(other, (float, int, complex)):
            other = PauliSum.from_pauli_strings(
                [PauliString(coefficient=other)])
        if isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])

        if not isinstance(other, PauliSum):
            return NotImplemented

        self._linear_dict -= other._linear_dict
        return self

    def __sub__(self, other):
        if not isinstance(other, (float, int, complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result -= other
        return result

    def __neg__(self):
        factory = type(self)
        return factory(-self._linear_dict)

    def __imul__(self, a: value.Scalar):
        self._linear_dict *= a
        return self

    def __mul__(self, a: value.Scalar):
        result = self.copy()
        result *= a
        return result

    def __rmul__(self, a: value.Scalar):
        return self.__mul__(a)

    def __truediv__(self, a: value.Scalar):
        return self.__mul__(1 / a)

    def __bool__(self) -> bool:
        return bool(self._linear_dict)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return 'cirq.{}({!r})'.format(class_name, self._linear_dict)

    def __format__(self, format_spec: str) -> str:
        terms = [(_pauli_string_from_unit(v), self._linear_dict[v])
                 for v in self._linear_dict.keys()]
        return value.linear_dict._format_terms(terms=terms,
                                               format_spec=format_spec)

    def __str__(self):
        return self.__format__('.3f')
