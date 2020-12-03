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
from typing import (
    AbstractSet,
    Mapping,
    Optional,
    Tuple,
    Union,
    List,
    FrozenSet,
    DefaultDict,
    TYPE_CHECKING,
)
import numbers

import numpy as np

from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.value.linear_dict import _format_terms
from cirq._compat import deprecated, deprecated_parameter

if TYPE_CHECKING:
    import cirq

UnitPauliStringT = FrozenSet[Tuple[raw_types.Qid, pauli_gates.Pauli]]
PauliSumLike = Union[
    int, float, complex, PauliString, 'PauliSum', pauli_string.SingleQubitPauliStringGateOperation
]
document(
    PauliSumLike,  # type: ignore
    """Any value that can be easily translated into a sum of Pauli products.
    """,
)


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

    def _is_compatible(self, gate: 'cirq.Gate') -> bool:
        return self.num_qubits() is None or self.num_qubits() == gate.num_qubits()

    def __add__(
        self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']
    ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__add__(other)

    def __iadd__(
        self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']
    ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__iadd__(other)

    def __sub__(
        self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']
    ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__sub__(other)

    def __isub__(
        self, other: Union[raw_types.Gate, 'LinearCombinationOfGates']
    ) -> 'LinearCombinationOfGates':
        if not isinstance(other, LinearCombinationOfGates):
            other = other.wrap_in_linear_combination()
        return super().__isub__(other)

    def __pow__(self, exponent: int) -> 'LinearCombinationOfGates':
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return NotImplemented
        if self.num_qubits() != 1:
            return NotImplemented
        pauli_basis = {
            identity.I,
            pauli_gates.X,
            pauli_gates.Y,
            pauli_gates.Z,
        }
        if not set(self.keys()).issubset(pauli_basis):
            return NotImplemented

        ai = self[identity.I]
        ax = self[pauli_gates.X]
        ay = self[pauli_gates.Y]
        az = self[pauli_gates.Z]
        bi, bx, by, bz = operator_spaces.pow_pauli_combination(ai, ax, ay, az, exponent)
        return LinearCombinationOfGates(
            {identity.I: bi, pauli_gates.X: bx, pauli_gates.Y: by, pauli_gates.Z: bz}
        )

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(gate) for gate in self.keys())

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for gate in self.keys() for name in protocols.parameter_names(gate)}

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolverOrSimilarType', recursive: bool
    ) -> 'LinearCombinationOfGates':
        return self.__class__(
            {
                protocols.resolve_parameters(gate, resolver, recursive): coeff
                for gate, coeff in self.items()
            }
        )

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying gates.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        if self._is_parameterized_():
            return NotImplemented
        num_qubits = self.num_qubits()
        if num_qubits is None:
            raise ValueError('Unknown number of qubits')
        num_dim = 2 ** num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for gate, coefficient in self.items():
            result += protocols.unitary(gate) * coefficient
        return result

    def _has_unitary_(self) -> bool:
        m = self.matrix()
        return m is not NotImplemented and linalg.is_unitary(m)

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if m is NotImplemented or linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

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

    def __init__(self, terms: Mapping[raw_types.Operation, value.Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of gate operations to coefficients in the linear
                combination being initialized.
        """
        super().__init__(terms, validator=self._is_compatible)

    def _is_compatible(self, operation: 'cirq.Operation') -> bool:
        return isinstance(operation, raw_types.Operation)

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """Returns qubits acted on self."""
        if not self:
            return ()
        qubit_sets = [set(op.qubits) for op in self.keys()]
        all_qubits = set.union(*qubit_sets)
        return tuple(sorted(all_qubits))

    def __pow__(self, exponent: int) -> 'LinearCombinationOfOperations':
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return NotImplemented
        if len(self.qubits) != 1:
            return NotImplemented
        qubit = self.qubits[0]
        i = identity.I(qubit)
        x = pauli_gates.X(qubit)
        y = pauli_gates.Y(qubit)
        z = pauli_gates.Z(qubit)
        pauli_basis = {i, x, y, z}
        if not set(self.keys()).issubset(pauli_basis):
            return NotImplemented

        ai, ax, ay, az = self[i], self[x], self[y], self[z]
        bi, bx, by, bz = operator_spaces.pow_pauli_combination(ai, ax, ay, az, exponent)
        return LinearCombinationOfOperations({i: bi, x: bx, y: by, z: bz})

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(op) for op in self.keys())

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for op in self.keys() for name in protocols.parameter_names(op)}

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolverOrSimilarType', recursive: bool
    ) -> 'LinearCombinationOfOperations':
        return self.__class__(
            {
                protocols.resolve_parameters(op, resolver, recursive): coeff
                for op, coeff in self.items()
            }
        )

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying operations.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        if self._is_parameterized_():
            return NotImplemented
        num_qubits = len(self.qubits)
        num_dim = 2 ** num_qubits
        qubit_to_axis = {q: i for i, q in enumerate(self.qubits)}
        result = np.zeros((2,) * (2 * num_qubits), dtype=np.complex128)
        for op, coefficient in self.items():
            identity = np.eye(num_dim, dtype=np.complex128).reshape(result.shape)
            workspace = np.empty_like(identity)
            axes = tuple(qubit_to_axis[q] for q in op.qubits)
            u = protocols.apply_unitary(op, protocols.ApplyUnitaryArgs(identity, workspace, axes))
            result += coefficient * u
        return result.reshape((num_dim, num_dim))

    def _has_unitary_(self) -> bool:
        m = self.matrix()
        return m is not NotImplemented and linalg.is_unitary(m)

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if m is NotImplemented or linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        """Computes Pauli expansion of self from Pauli expansions of terms."""

        def extend_term(
            pauli_names: str, qubits: Tuple['cirq.Qid', ...], all_qubits: Tuple['cirq.Qid', ...]
        ) -> str:
            """Extends Pauli product on qubits to product on all_qubits."""
            assert len(pauli_names) == len(qubits)
            qubit_to_pauli_name = dict(zip(qubits, pauli_names))
            return ''.join(qubit_to_pauli_name.get(q, 'I') for q in all_qubits)

        def extend(
            expansion: value.LinearDict[str],
            qubits: Tuple['cirq.Qid', ...],
            all_qubits: Tuple['cirq.Qid', ...],
        ) -> value.LinearDict[str]:
            """Extends Pauli expansion on qubits to expansion on all_qubits."""
            return value.LinearDict(
                {extend_term(p, qubits, all_qubits): c for p, c in expansion.items()}
            )

        result = value.LinearDict({})  # type: value.LinearDict[str]
        for op, coefficient in self.items():
            expansion = protocols.pauli_expansion(op)
            extended_expansion = extend(expansion, op.qubits, self.qubits)
            result += extended_expansion * coefficient
        return result


def _is_linear_dict_of_unit_pauli_string(linear_dict: value.LinearDict[UnitPauliStringT]) -> bool:
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


def _pauli_string_from_unit(unit: UnitPauliStringT, coefficient: Union[int, float, complex] = 1):
    return PauliString(qubit_pauli_map=dict(unit), coefficient=coefficient)


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

    def __init__(self, linear_dict: Optional[value.LinearDict[UnitPauliStringT]] = None):
        if linear_dict is None:
            linear_dict = value.LinearDict()
        if not _is_linear_dict_of_unit_pauli_string(linear_dict):
            raise ValueError(
                "PauliSum constructor takes a LinearDict[UnitPauliStringT]. "
                "Consider using PauliSum.from_pauli_strings() or adding and "
                "subtracting PauliStrings"
            )
        self._linear_dict = linear_dict

    def _value_equality_values_(self):
        return self._linear_dict

    @staticmethod
    def wrap(val: PauliSumLike) -> 'PauliSum':
        if isinstance(val, PauliSum):
            return val
        return PauliSum() + val

    @classmethod
    def from_pauli_strings(cls, terms: Union[PauliString, List[PauliString]]) -> 'PauliSum':
        if isinstance(terms, PauliString):
            terms = [terms]
        termdict: DefaultDict[UnitPauliStringT, value.Scalar] = defaultdict(lambda: 0)
        for pstring in terms:
            key = frozenset(pstring._qubit_pauli_map.items())
            termdict[key] += pstring.coefficient
        return cls(linear_dict=value.LinearDict(termdict))

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        qs = {q for k in self._linear_dict.keys() for q, _ in k}
        return tuple(sorted(qs))

    def copy(self) -> 'PauliSum':
        factory = type(self)
        return factory(self._linear_dict.copy())

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self from underlying Pauli operations.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """
        num_qubits = len(self.qubits)
        num_dim = 2 ** num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for vec, coeff in self._linear_dict.items():
            op = _pauli_string_from_unit(vec)
            result += coeff * op.matrix(self.qubits)
        return result

    def _has_unitary_(self) -> bool:
        return linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    @deprecated(deadline='v0.10.0', fix='Use expectation_from_state_vector instead.')
    def expectation_from_wavefunction(
        self,
        state: np.ndarray,
        qubit_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        return self.expectation_from_state_vector(
            state_vector=state,
            qubit_map=qubit_map,
            atol=atol,
            check_preconditions=check_preconditions,
        )

    @deprecated_parameter(
        deadline='v0.10.0',
        fix='Use state_vector instead',
        parameter_desc='state',
        match=lambda args, kwargs: 'state' in kwargs,
        rewrite=lambda args, kwargs: (
            args,
            {('state_vector' if k == 'state' else k): v for k, v in kwargs.items()},
        ),
    )
    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qubit_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        """Evaluate the expectation of this PauliSum given a state vector.

        See `PauliString.expectation_from_state_vector`.

        Args:
            state: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliSum to the
                indices of the qubits that `state_vector` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state_vector` represents
                a valid state vector.

        Returns:
            The expectation value of the input state.
        """
        if any(abs(p.coefficient.imag) > 0.0001 for p in self):
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                "PauliString <{}>. Coefficient must be real.".format(self)
            )

        # TODO: Avoid enforce specific complex type. This is necessary to
        # prevent an `apply_unitary` bug.
        # Github issue: https://github.com/quantumlib/Cirq/issues/2041
        if state_vector.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or np.complex128")

        size = state_vector.size
        num_qubits = size.bit_length() - 1
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)

        if len(state_vector.shape) != 1 and state_vector.shape != (2,) * num_qubits:
            raise ValueError(
                "Input array does not represent a state vector "
                "with shape `(2 ** n,)` or `(2, ..., 2)`."
            )

        if check_preconditions:
            qis.validate_normalized_state_vector(
                state_vector=state_vector,
                qid_shape=(2,) * num_qubits,
                dtype=state_vector.dtype,
                atol=atol,
            )
        return sum(
            p._expectation_from_state_vector_no_validation(state_vector, qubit_map) for p in self
        )

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qubit_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        """Evaluate the expectation of this PauliSum given a density matrix.

        See `PauliString.expectation_from_density_matrix`.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliSum to the
                indices of the qubits that `state` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid density matrix.

        Returns:
            The expectation value of the input state.
        """
        if any(abs(p.coefficient.imag) > 0.0001 for p in self):
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                "PauliString <{}>. Coefficient must be real.".format(self)
            )

        # FIXME: Avoid enforce specific complex type. This is necessary to
        # prevent an `apply_unitary` bug (Issue #2041).
        if state.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or np.complex128")

        size = state.size
        num_qubits = int(np.sqrt(size)).bit_length() - 1
        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)

        dim = int(np.sqrt(size))
        if state.shape != (dim, dim) and state.shape != (2, 2) * num_qubits:
            raise ValueError(
                "Input array does not represent a density matrix "
                "with shape `(2 ** n, 2 ** n)` or `(2, ..., 2)`."
            )

        if check_preconditions:
            # Do not enforce reshaping if the state all axes are dimension 2.
            _ = qis.to_valid_density_matrix(
                density_matrix_rep=state.reshape(dim, dim),
                num_qubits=num_qubits,
                dtype=state.dtype,
                atol=atol,
            )
        return sum(p._expectation_from_density_matrix_no_validation(state, qubit_map) for p in self)

    def __iter__(self):
        for vec, coeff in self._linear_dict.items():
            yield _pauli_string_from_unit(vec, coeff)

    def __len__(self) -> int:
        return len(self._linear_dict)

    def __iadd__(self, other):
        if isinstance(other, numbers.Complex):
            other = PauliSum.from_pauli_strings([PauliString(coefficient=other)])
        elif isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])

        if not isinstance(other, PauliSum):
            return NotImplemented

        self._linear_dict += other._linear_dict
        return self

    def __add__(self, other):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result += other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __isub__(self, other):
        if isinstance(other, numbers.Complex):
            other = PauliSum.from_pauli_strings([PauliString(coefficient=other)])
        if isinstance(other, PauliString):
            other = PauliSum.from_pauli_strings([other])

        if not isinstance(other, PauliSum):
            return NotImplemented

        self._linear_dict -= other._linear_dict
        return self

    def __sub__(self, other):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result -= other
        return result

    def __neg__(self):
        factory = type(self)
        return factory(-self._linear_dict)

    def __imul__(self, other: PauliSumLike):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        if isinstance(other, numbers.Complex):
            self._linear_dict *= other
        elif isinstance(other, PauliString):
            temp = PauliSum.from_pauli_strings([term * other for term in self])
            self._linear_dict = temp._linear_dict
        elif isinstance(other, PauliSum):
            temp = PauliSum.from_pauli_strings(
                [term * other_term for term in self for other_term in other]
            )
            self._linear_dict = temp._linear_dict

        return self

    def __mul__(self, other: PauliSumLike):
        if not isinstance(other, (numbers.Complex, PauliString, PauliSum)):
            return NotImplemented
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: PauliSumLike):
        if isinstance(other, numbers.Complex):
            result = self.copy()
            result *= other
            return result
        elif isinstance(other, PauliString):
            result = self.copy()
            return PauliSum.from_pauli_strings([other]) * result
        return NotImplemented

    def __pow__(self, exponent: int):
        if not isinstance(exponent, numbers.Integral):
            return NotImplemented
        if exponent == 0:
            return PauliSum(value.LinearDict({frozenset(): 1 + 0j}))
        if exponent > 0:
            base = self.copy()
            for _ in range(exponent - 1):
                base *= base
            return base
        return NotImplemented

    def __truediv__(self, a: value.Scalar):
        return self.__mul__(1 / a)

    def __bool__(self) -> bool:
        return bool(self._linear_dict)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'cirq.{class_name}({self._linear_dict!r})'

    def __format__(self, format_spec: str) -> str:
        terms = [
            (_pauli_string_from_unit(v), self._linear_dict[v]) for v in self._linear_dict.keys()
        ]
        return _format_terms(terms=terms, format_spec=format_spec)

    def __str__(self) -> str:
        return self.__format__('.3f')
