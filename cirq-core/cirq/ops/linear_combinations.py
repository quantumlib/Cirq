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

import numbers
from collections import defaultdict
from typing import (
    AbstractSet,
    Any,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
from scipy.sparse import csr_matrix
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Not, Or, Xor

from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, pauli_gates, pauli_string, raw_types
from cirq.ops.pauli_string import _validate_qubit_mapping, PauliString
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms

if TYPE_CHECKING:
    import cirq

UnitPauliStringT = FrozenSet[Tuple[raw_types.Qid, pauli_gates.Pauli]]
PauliSumLike = Union[
    complex, PauliString, 'PauliSum', pauli_string.SingleQubitPauliStringGateOperation
]
document(
    PauliSumLike,
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

    def __init__(self, terms: Mapping[raw_types.Gate, 'cirq.TParamValComplex']) -> None:
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
        pauli_basis = {identity.I, pauli_gates.X, pauli_gates.Y, pauli_gates.Z}
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
        return any(protocols.is_parameterized(item) for item in self.items())

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for item in self.items() for name in protocols.parameter_names(item)}

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'LinearCombinationOfGates':
        return self.__class__(
            {
                protocols.resolve_parameters(
                    gate, resolver, recursive
                ): protocols.resolve_parameters(coeff, resolver, recursive)
                for gate, coeff in self.items()
            }
        )

    def matrix(self) -> np.ndarray:
        """Reconstructs matrix of self using unitaries of underlying gates.

        Raises:
            ValueError: If the number of qubits has not been specified.
        """
        if self._is_parameterized_():
            return NotImplemented
        num_qubits = self.num_qubits()
        if num_qubits is None:
            raise ValueError('Unknown number of qubits')
        num_dim = 2**num_qubits
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
        result: value.LinearDict[str] = value.LinearDict({})
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

    def __init__(self, terms: Mapping[raw_types.Operation, 'cirq.TParamValComplex']) -> None:
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
        return any(protocols.is_parameterized(item) for item in self.items())

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for item in self.items() for name in protocols.parameter_names(item)}

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'LinearCombinationOfOperations':
        return self.__class__(
            {
                protocols.resolve_parameters(op, resolver, recursive): protocols.resolve_parameters(
                    coeff, resolver, recursive
                )
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
        num_dim = 2**num_qubits
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

        result: value.LinearDict[str] = value.LinearDict({})
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


def _pauli_string_from_unit(
    unit: UnitPauliStringT, coefficient: Union[int, float, 'cirq.TParamValComplex'] = 1
):
    return PauliString(qubit_pauli_map=dict(unit), coefficient=coefficient)


@value.value_equality(approximate=True, unhashable=True)
class PauliSum:
    """Represents operator defined by linear combination of PauliStrings.

    Since `cirq.PauliString`s store their own coefficients, this class
    does not implement the `cirq.LinearDict` interface. Instead, you can
    add and subtract terms and then iterate over the resulting
    (simplified) expression.

    Under the hood, this class is backed by a LinearDict with coefficient-less
    PauliStrings as keys. PauliStrings are reconstructed on-the-fly during
    iteration.

    PauliSums can be constructed explicitly:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.PauliSum.from_pauli_strings([
    ...     cirq.PauliString(-1, cirq.X(a), cirq.Y(b)),
    ...     cirq.PauliString(2, cirq.Z(a), cirq.Z(b)),
    ...     cirq.PauliString(0.5, cirq.Y(a), cirq.Y(b))
    ... ])
    >>> print(psum)
    -1.000*X(q(0, 0))*Y(q(0, 1))+2.000*Z(q(0, 0))*Z(q(0, 1))+0.500*Y(q(0, 0))*Y(q(0, 1))


    or implicitly:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.X(a) * cirq.X(b) + 3.0 * cirq.Y(a)
    >>> print(psum)
    1.000*X(q(0, 0))*X(q(0, 1))+3.000*Y(q(0, 0))

    basic arithmetic and expectation operations are supported as well:


    >>> a, b = cirq.GridQubit.rect(1, 2)
    >>> psum = cirq.X(a) * cirq.X(b) + 3.0 * cirq.Y(a)
    >>> two_psum = 2 * psum
    >>> four_psum = two_psum + two_psum
    >>> print(four_psum)
    4.000*X(q(0, 0))*X(q(0, 1))+12.000*Y(q(0, 0))


    >>> expectation = four_psum.expectation_from_state_vector(
    ...     np.array([0.707106, 0, 0, 0.707106], dtype=complex),
    ...     qubit_map={a: 0, b: 1}
    ... )
    >>> print(f'{expectation:.1f}')
    4.0+0.0j
    """

    def __init__(self, linear_dict: Optional[value.LinearDict[UnitPauliStringT]] = None):
        """Construct a PauliSum from a linear dictionary.

        Note, the preferred method of constructing PauliSum objects is either implicitly
        or via the `from_pauli_strings` function.

        Args:
            linear_dict: Set of  (`cirq.Qid`, `cirq.Pauli`) tuples to construct the sum
                from.

        Raises:
            ValueError: If structure of `linear_dict` contains tuples other than the
                form (`cirq.Qid`, `cirq.Pauli`).
        """
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
        """Convert a `cirq.PauliSumLike` object to a PauliSum

        Attempts to convert an existing int, float, complex, `cirq.PauliString`,
        `cirq.PauliSum` or `cirq.SingleQubitPauliStringGateOperation` into
        a `cirq.PauliSum` object. For example:


        >>> my_psum = cirq.PauliSum.wrap(2.345)
        >>> my_psum
        cirq.PauliSum(cirq.LinearDict({frozenset(): (2.345+0j)}))


        Args:
            `cirq.PauliSumLike` to convert to PauliSum.

        Returns:
            PauliSum representation of `val`.
        """
        if isinstance(val, PauliSum):
            return val
        return PauliSum() + val

    @classmethod
    def from_pauli_strings(cls, terms: Union[PauliString, List[PauliString]]) -> 'PauliSum':
        """Returns a PauliSum by combining `cirq.PauliString` terms.

        Args:
            terms: `cirq.PauliString` or List of `cirq.PauliString`s to use inside
                of this PauliSum object.
        Returns:
            PauliSum object representing the addition of all the `cirq.PauliString`
                terms in `terms`.
        """
        if isinstance(terms, PauliString):
            terms = [terms]
        termdict: DefaultDict[UnitPauliStringT, value.Scalar] = defaultdict(lambda: 0)
        for pstring in terms:
            key = frozenset(pstring._qubit_pauli_map.items())
            termdict[key] += pstring.coefficient
        return cls(linear_dict=value.LinearDict(termdict))

    @classmethod
    def from_boolean_expression(
        cls, boolean_expr: Expr, qubit_map: Dict[str, 'cirq.Qid']
    ) -> 'PauliSum':
        """Builds the Hamiltonian representation of a Boolean expression.

        This is based on "On the representation of Boolean and real functions as Hamiltonians for
        quantum computing" by Stuart Hadfield, https://arxiv.org/abs/1804.09130

        Args:
            boolean_expr: A Sympy expression containing symbols and Boolean operations
            qubit_map: map of string (boolean variable name) to qubit.

        Return:
            The PauliSum that represents the Boolean expression.

        Raises:
            ValueError: If `boolean_expr` is of an unsupported type.
        """
        if isinstance(boolean_expr, Symbol):
            # In table 1, the entry for 'x' is '1/2.I - 1/2.Z'
            return cls.from_pauli_strings(
                [
                    PauliString({}, 0.5),
                    PauliString({qubit_map[boolean_expr.name]: pauli_gates.Z}, -0.5),
                ]
            )

        if isinstance(boolean_expr, (And, Not, Or, Xor)):
            sub_pauli_sums = [
                cls.from_boolean_expression(sub_boolean_expr, qubit_map)
                for sub_boolean_expr in boolean_expr.args
            ]
            # We apply the equalities of theorem 1.
            if isinstance(boolean_expr, And):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum * sub_pauli_sum
            elif isinstance(boolean_expr, Not):
                assert len(sub_pauli_sums) == 1
                pauli_sum = cls.from_pauli_strings(PauliString({}, 1.0)) - sub_pauli_sums[0]
            elif isinstance(boolean_expr, Or):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum + sub_pauli_sum - pauli_sum * sub_pauli_sum
            elif isinstance(boolean_expr, Xor):
                pauli_sum = cls.from_pauli_strings(PauliString({}, 0.0))
                for sub_pauli_sum in sub_pauli_sums:
                    pauli_sum = pauli_sum + sub_pauli_sum - 2.0 * pauli_sum * sub_pauli_sum
            return pauli_sum

        raise ValueError(f'Unsupported type: {type(boolean_expr)}')

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """The sorted list of qubits used in this PauliSum."""
        qs = {q for k in self._linear_dict.keys() for q, _ in k}
        return tuple(sorted(qs))

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliSum':
        """Return a new PauliSum on `new_qubits`.

        Args:
            *new_qubits: `cirq.Qid` objects to replace existing
                qubit objects in this PauliSum.

        Returns:
            PauliSum with new_qubits replacing the previous
                qubits.

        Raises:
            ValueError: If len(new_qubits) != len(self.qubits).

        """
        qubits = self.qubits
        if len(new_qubits) != len(qubits):
            raise ValueError('Incorrect number of qubits for PauliSum.')
        qubit_map = dict(zip(qubits, new_qubits))
        new_pauli_strings = []
        for pauli_string in self:
            new_pauli_strings.append(pauli_string.map_qubits(qubit_map))
        return PauliSum.from_pauli_strings(new_pauli_strings)

    def copy(self) -> 'PauliSum':
        """Return a copy of this PauliSum.

        Returns: A copy of this PauliSum.
        """
        factory = type(self)
        return factory(self._linear_dict.copy())

    def matrix(self, qubits: Optional[Iterable[raw_types.Qid]] = None) -> np.ndarray:
        """Returns the matrix of this PauliSum in computational basis of qubits.

        Args:
            qubits: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the Pauli sum is to
                be computed. If none is provided the default ordering of
                `self.qubits` is used.  Qubits present in `qubits` but absent from
                `self.qubits` are acted on by the identity.

        Returns:
            np.ndarray representing the matrix of this PauliSum expression.

        Raises:
            TypeError: if any of the gates in self does not provide a unitary.
        """

        qubits = self.qubits if qubits is None else tuple(qubits)
        num_qubits = len(qubits)
        num_dim = 2**num_qubits
        result = np.zeros((num_dim, num_dim), dtype=np.complex128)
        for vec, coeff in self._linear_dict.items():
            op = _pauli_string_from_unit(vec)
            result += coeff * op.matrix(qubits)
        return result

    def _has_unitary_(self) -> bool:
        return linalg.is_unitary(self.matrix())

    def _unitary_(self) -> np.ndarray:
        m = self.matrix()
        if linalg.is_unitary(m):
            return m
        raise ValueError(f'{self} is not unitary')

    def _json_dict_(self):
        def key_json(k: UnitPauliStringT):
            return [list(e) for e in sorted(k)]

        return {'items': list((key_json(k), v) for k, v in self._linear_dict.items())}

    @classmethod
    def _from_json_dict_(cls, items, **kwargs):
        mapping = {
            frozenset(tuple(qid_pauli) for qid_pauli in unit_pauli_string): val
            for unit_pauli_string, val in items
        }
        return cls(linear_dict=value.LinearDict(mapping))

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
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliSum to the
                indices of the qubits that `state_vector` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state_vector` represents
                a valid state vector.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If any of the coefficients are imaginary,
                so that this is not Hermitian.
            TypeError: If the input state is not a complex type.
            ValueError: If the input vector is not the correct size or shape.
        """
        if any(abs(p.coefficient.imag) > 0.0001 for p in self):
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                f"PauliString <{self}>. Coefficient must be real."
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

        Raises:
            NotImplementedError: If any of the coefficients are imaginary,
                so that this is not Hermitian.
            TypeError: If the input state is not a complex type.
            ValueError: If the input vector is not the correct size or shape.
        """
        if any(abs(p.coefficient.imag) > 0.0001 for p in self):
            raise NotImplementedError(
                "Cannot compute expectation value of a non-Hermitian "
                f"PauliString <{self}>. Coefficient must be real."
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
            if hasattr(other, 'gate') and isinstance(other.gate, identity.IdentityGate):
                other = PauliString(other)
            else:
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
            self._linear_dict *= complex(other)
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
            result = self.copy()
            for _ in range(exponent - 1):
                result *= self
            return result
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


def _projector_string_from_projector_dict(projector_dict, coefficient=1.0):
    return ProjectorString(dict(projector_dict), coefficient)


@value.value_equality(approximate=True, unhashable=True)
class ProjectorSum:
    """List of mappings representing a sum of projector operators."""

    def __init__(
        self, linear_dict: Optional[value.LinearDict[FrozenSet[Tuple[raw_types.Qid, int]]]] = None
    ):
        """Constructor for ProjectorSum

        Args:
            linear_dict: A linear dictionary from a set of tuples of (Qubit, integer) to a complex
                number. The tuple is a projector onto the qubit and the complex number is the
                weight of these projections.
        """
        self._linear_dict: value.LinearDict[FrozenSet[Tuple[raw_types.Qid, int]]] = (
            linear_dict if linear_dict is not None else value.LinearDict({})
        )

    def _value_equality_values_(self):
        return self._linear_dict

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        qs = {q for k in self._linear_dict.keys() for q, _ in k}
        return tuple(sorted(qs))

    def _json_dict_(self) -> Dict[str, Any]:
        linear_dict = []
        for projector_dict, scalar in dict(self._linear_dict).items():
            key = [[k, v] for k, v in dict(projector_dict).items()]
            linear_dict.append([key, scalar])
        return {'linear_dict': linear_dict}

    @classmethod
    def _from_json_dict_(cls, linear_dict, **kwargs):
        converted_dict = {}
        for projector_string in linear_dict:
            projector_dict = {x[0]: x[1] for x in projector_string[0]}
            scalar = projector_string[1]
            key = frozenset(projector_dict.items())
            converted_dict[key] = scalar
        return cls(linear_dict=value.LinearDict(converted_dict))

    @classmethod
    def from_projector_strings(
        cls, terms: Union[ProjectorString, List[ProjectorString]]
    ) -> 'ProjectorSum':
        """Builds a ProjectorSum from one or more ProjectorString(s).

        Args:
            terms: Either a single ProjectorString or a list of ProjectorStrings.

        Returns:
            A ProjectorSum.
        """
        if isinstance(terms, ProjectorString):
            terms = [terms]
        termdict: DefaultDict[FrozenSet[Tuple[raw_types.Qid, int]], value.Scalar] = defaultdict(
            lambda: 0.0
        )
        for pstring in terms:
            key = frozenset(pstring.projector_dict.items())
            termdict[key] += pstring.coefficient
        return cls(linear_dict=value.LinearDict(termdict))

    def copy(self) -> 'ProjectorSum':
        return ProjectorSum(self._linear_dict.copy())

    def matrix(self, projector_qids: Optional[Iterable[raw_types.Qid]] = None) -> csr_matrix:
        """Returns the matrix of self in computational basis of qubits.

        Args:
            projector_qids: Ordered collection of qubits that determine the subspace in which the
                matrix representation of the ProjectorSum is to be computed. Qbits absent from
                self.qubits are acted on by the identity. Defaults to the qubits of the
                projector_dict.

        Returns:
            A sparse matrix that is the projection in the specified basis.
        """
        return sum(
            coeff * _projector_string_from_projector_dict(vec).matrix(projector_qids)
            for vec, coeff in self._linear_dict.items()
        )

    def expectation_from_state_vector(
        self, state_vector: np.ndarray, qid_map: Mapping[raw_types.Qid, int]
    ) -> float:
        """Compute the expectation value of this ProjectorSum given a state vector.

        Projects the state vector onto the sum of projectors and computes the expectation of the
        measurements.

        Args:
            state_vector: An array representing a valid state vector.
            qid_map: A map from all qubits used in this ProjectorSum to the indices of the qubits
                that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
        return sum(
            coeff
            * _projector_string_from_projector_dict(vec).expectation_from_state_vector(
                state_vector, qid_map
            )
            for vec, coeff in self._linear_dict.items()
        )

    def expectation_from_density_matrix(
        self, state: np.ndarray, qid_map: Mapping[raw_types.Qid, int]
    ) -> float:
        """Expectation of the sum of projections from a density matrix.

        Projects the density matrix onto the sum of projectors and computes the expectation of the
        measurements.

        Args:
            state: An array representing a valid  density matrix.
            qid_map: A map from all qubits used in this ProjectorSum to the indices of the qubits
                that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
        return sum(
            coeff
            * _projector_string_from_projector_dict(vec).expectation_from_density_matrix(
                state, qid_map
            )
            for vec, coeff in self._linear_dict.items()
        )

    def __iter__(self):
        for vec, coeff in self._linear_dict.items():
            yield _projector_string_from_projector_dict(vec, coeff)

    def __len__(self) -> int:
        return len(self._linear_dict)

    def __truediv__(self, a: value.Scalar):
        return self.__mul__(1 / a)

    def __bool__(self) -> bool:
        return bool(self._linear_dict)

    def __iadd__(self, other: Union['ProjectorString', 'ProjectorSum']):
        if isinstance(other, ProjectorString):
            other = ProjectorSum.from_projector_strings(other)
        elif not isinstance(other, ProjectorSum):
            return NotImplemented
        self._linear_dict += other._linear_dict
        return self

    def __add__(self, other: Union['ProjectorString', 'ProjectorSum']):
        if isinstance(other, ProjectorString):
            other = ProjectorSum.from_projector_strings(other)
        elif not isinstance(other, ProjectorSum):
            return NotImplemented
        result = self.copy()
        result += other
        return result

    def __isub__(self, other: Union['ProjectorString', 'ProjectorSum']):
        if isinstance(other, ProjectorString):
            other = ProjectorSum.from_projector_strings(other)
        elif not isinstance(other, ProjectorSum):
            return NotImplemented
        self._linear_dict -= other._linear_dict
        return self

    def __sub__(self, other: Union['ProjectorString', 'ProjectorSum']):
        if isinstance(other, ProjectorString):
            other = ProjectorSum.from_projector_strings(other)
        elif not isinstance(other, ProjectorSum):
            return NotImplemented
        result = self.copy()
        result -= other
        return result

    def __neg__(self):
        factory = type(self)
        return factory(-self._linear_dict)

    def __imul__(self, other: value.Scalar):
        if not isinstance(other, numbers.Complex):
            return NotImplemented
        self._linear_dict *= other
        return self

    def __mul__(self, other: value.Scalar):
        if not isinstance(other, numbers.Complex):
            return NotImplemented
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: value.Scalar):
        if not isinstance(other, numbers.Complex):
            return NotImplemented
        result = self.copy()
        result *= other
        return result
