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
import cmath
import math
import numbers
from types import NotImplementedType
from typing import (
    Any,
    cast,
    Dict,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    overload,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
    ValuesView,
    AbstractSet,
    Callable,
    Generic,
)

import numpy as np
import sympy

from cirq import value, protocols, linalg, qis, _compat
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
    clifford_gate,
    common_gates,
    gate_operation,
    global_phase_op,
    identity,
    op_tree,
    pauli_gates,
    pauli_interaction_gate,
    raw_types,
    dense_pauli_string,
)

if TYPE_CHECKING:
    import cirq

# Lazy imports to break circular dependencies.
linear_combinations = LazyLoader("linear_combinations", globals(), "cirq.ops.linear_combinations")

TDefault = TypeVar('TDefault')
TKey = TypeVar('TKey', bound=raw_types.Qid)
TKeyNew = TypeVar('TKeyNew', bound=raw_types.Qid)
TKeyOther = TypeVar('TKeyOther', bound=raw_types.Qid)

# A value that can be unambiguously converted into a `cirq.PauliString`.

PAULI_STRING_LIKE = Union[
    complex,
    'cirq.OP_TREE',
    Mapping[TKey, 'cirq.PAULI_GATE_LIKE'],
    Iterable,  # of PAULI_STRING_LIKE, but mypy doesn't do recursive types yet.
]
document(
    PAULI_STRING_LIKE,
    """A `cirq.PauliString` or a value that can easily be converted into one.

    Complex numbers turn into the coefficient of an empty Pauli string.

    Dictionaries from qubit to Pauli operation are wrapped into a Pauli string.
    Each Pauli operation can be specified as a cirq object (e.g. `cirq.X`) or as
    a string (e.g. `"X"`) or as an integer where 0=I, 1=X, 2=Y, 3=Z.

    Collections of Pauli operations are recursively multiplied into a single
    Pauli string.
    """,
)

PAULI_GATE_LIKE = Union['cirq.Pauli', 'cirq.IdentityGate', str, int]
document(
    PAULI_GATE_LIKE,
    """An object that can be interpreted as a Pauli gate.

    Allowed values are:

    1. Cirq gates: `cirq.I`, `cirq.X`, `cirq.Y`, `cirq.Z`.
    2. Strings: "I", "X", "Y", "Z". Equivalently "i", "x", "y", "z".
    3. Integers from 0 to 3, with the convention 0=I, 1=X, 2=Y, 3=Z.
    """,
)


@value.value_equality(approximate=True, manual_cls=True)
class PauliString(raw_types.Operation, Generic[TKey]):
    """Represents a multi-qubit pauli operator or pauli observable.

    `cirq.PauliString` represents a multi-qubit pauli operator, i.e.
    a tensor product of single qubit (non identity) pauli operations,
    each acting on a different qubit. For  example,

    - X(0) * Y(1) * Z(2): Represents a pauli string which is a tensor product of
                          `cirq.X(q0)`, `cirq.Y(q1)` and `cirq.Z(q2)`.

    If more than one pauli operation acts on the same set of qubits, their composition is
    immediately reduced to an equivalent (possibly multi-qubit) Pauli operator. Also, identity
    operations are dropped by the `PauliString` class. For example:

    >>> a, b = cirq.LineQubit.range(2)
    >>> print(cirq.X(a) * cirq.Y(b)) # Tensor product of Pauli's acting on different qubits.
    X(q(0))*Y(q(1))
    >>> print(cirq.X(a) * cirq.Y(a)) # Composition is reduced to an equivalent PauliString.
    1j*Z(q(0))
    >>> print(cirq.X(a) * cirq.I(b)) # Identity operations are dropped by default.
    X(q(0))
    >>> print(cirq.PauliString()) # String representation of an "empty" PaulString is "I".
    I

    `cirq.PauliString` is often used to represent:
    - Pauli operators: Can be inserted into circuits as multi qubit operations.
    - Pauli observables: Can be measured using either `cirq.measure_single_paulistring`/
                        `cirq.measure_paulistring_terms`; or using the observable
                        measurement framework in `cirq.measure_observables`.

    PauliStrings can be constructed via various different ways, some examples are
    given as follows:

    >>> a, b, c = cirq.LineQubit.range(3)
    >>> print(cirq.PauliString([cirq.X(a), cirq.X(a)]))
    I
    >>> print(cirq.PauliString(-1, cirq.X(a), cirq.Y(b), cirq.Z(c)))
    -X(q(0))*Y(q(1))*Z(q(2))
    >>> print(-1 * cirq.X(a) * cirq.Y(b) * cirq.Z(c))
    -X(q(0))*Y(q(1))*Z(q(2))
    >>> print(cirq.PauliString({a: cirq.X}, [-2, 3, cirq.Y(a)]))
    -6j*Z(q(0))
    >>> print(cirq.PauliString({a: cirq.I, b: cirq.X}))
    X(q(1))
    >>> print(cirq.PauliString({a: cirq.Y}, qubit_pauli_map={a: cirq.X}))
    1j*Z(q(0))

    Note that `cirq.PauliString`s are immutable objects. If you need a mutable version
    of pauli strings, see `cirq.MutablePauliString`.
    """

    def __init__(
        self,
        *contents: 'cirq.PAULI_STRING_LIKE',
        qubit_pauli_map: Optional[Dict[TKey, 'cirq.Pauli']] = None,
        coefficient: 'cirq.TParamValComplex' = 1,
    ):
        """Initializes a new `PauliString` operation.

        Args:
            *contents: A value or values to convert into a pauli string. This
                can be a number, a pauli operation, a dictionary from qubit to
                pauli/identity gates, or collections thereof. If a list of
                values is given, they are each individually converted and then
                multiplied from left to right in order.
            qubit_pauli_map: Initial dictionary mapping qubits to pauli
                operations. Defaults to the empty dictionary. Note that, unlike
                dictionaries passed to contents, this dictionary must not
                contain any identity gate values. Further note that this
                argument specifies values that are logically *before* factors
                specified in `contents`; `contents` are *right* multiplied onto
                the values in this dictionary.
            coefficient: Initial scalar coefficient or symbol. Defaults to 1.

        Raises:
            TypeError: If the `qubit_pauli_map` has values that are not Paulis.
        """
        if _compat.__cirq_debug__.get() and qubit_pauli_map is not None:
            for v in qubit_pauli_map.values():
                if not isinstance(v, pauli_gates.Pauli):
                    raise TypeError(f'{v} is not a Pauli')

        self._qubit_pauli_map: Dict[TKey, 'cirq.Pauli'] = qubit_pauli_map or {}
        self._coefficient: Union['cirq.TParamValComplex', sympy.Expr] = (
            coefficient if isinstance(coefficient, sympy.Expr) else complex(coefficient)
        )
        if contents:
            m = self.mutable_copy().inplace_left_multiply_by(contents).frozen()
            self._qubit_pauli_map = m._qubit_pauli_map
            self._coefficient = m._coefficient

    @property
    def coefficient(self) -> 'cirq.TParamValComplex':
        """A scalar coefficient or symbol."""
        return self._coefficient

    def _value_equality_values_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            q, p = list(self._qubit_pauli_map.items())[0]
            return gate_operation.GateOperation(p, [q])._value_equality_values_()

        return (frozenset(self._qubit_pauli_map.items()), self._coefficient)

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            # JSON requires mappings to have string keys.
            'qubit_pauli_map': list(self._qubit_pauli_map.items()),
            'coefficient': self.coefficient,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_pauli_map, coefficient, **kwargs):
        return cls(qubit_pauli_map=dict(qubit_pauli_map), coefficient=coefficient)

    def _value_equality_values_cls_(self):
        if len(self._qubit_pauli_map) == 1 and self.coefficient == 1:
            return gate_operation.GateOperation
        return PauliString

    def equal_up_to_coefficient(self, other: 'cirq.PauliString') -> bool:
        """Returns true of `self` and `other` are equal pauli strings, ignoring the coefficient."""
        return self._qubit_pauli_map == other._qubit_pauli_map

    def __getitem__(self, key: TKey) -> pauli_gates.Pauli:
        return self._qubit_pauli_map[key]

    # pylint: disable=function-redefined
    @overload
    def get(self, key: Any, default: None = None) -> Optional[pauli_gates.Pauli]:
        pass

    @overload
    def get(self, key: Any, default: TDefault) -> Union[pauli_gates.Pauli, TDefault]:
        pass

    def get(
        self, key: Any, default: Optional[TDefault] = None
    ) -> Union[pauli_gates.Pauli, TDefault, None]:
        """Returns the `cirq.Pauli` operation acting on qubit `key` or `default` if none exists."""
        return self._qubit_pauli_map.get(key, default)

    @overload
    def __mul__(
        self, other: 'cirq.PauliString[TKeyOther]'
    ) -> 'cirq.PauliString[Union[TKey, TKeyOther]]':
        pass

    @overload
    def __mul__(
        self, other: Mapping[TKeyOther, 'cirq.PAULI_GATE_LIKE']
    ) -> 'cirq.PauliString[Union[TKey, TKeyOther]]':
        pass

    @overload
    def __mul__(
        self, other: Iterable['cirq.PAULI_STRING_LIKE']
    ) -> 'cirq.PauliString[Union[TKey, cirq.Qid]]':
        pass

    @overload
    def __mul__(self, other: 'cirq.Operation') -> 'cirq.PauliString[Union[TKey, cirq.Qid]]':
        pass

    @overload
    def __mul__(self, other: complex) -> 'cirq.PauliString[TKey]':
        pass

    def __mul__(self, other):
        known = False
        if isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
            known = True
        elif isinstance(other, (PauliString, numbers.Number)):
            known = True
        if known:
            return PauliString(
                cast(PAULI_STRING_LIKE, other),
                qubit_pauli_map=self._qubit_pauli_map,
                coefficient=self.coefficient,
            )
        return NotImplemented

    # pylint: enable=function-redefined

    @property
    def gate(self) -> 'cirq.DensePauliString':
        """Returns a `cirq.DensePauliString`"""
        order: List[Optional[pauli_gates.Pauli]] = [
            None,
            pauli_gates.X,
            pauli_gates.Y,
            pauli_gates.Z,
        ]
        from cirq.ops.dense_pauli_string import DensePauliString

        return DensePauliString(
            coefficient=self.coefficient, pauli_mask=[order.index(self[q]) for q in self.qubits]
        )

    def __rmul__(self, other) -> 'PauliString':
        if isinstance(other, numbers.Complex):
            return PauliString(
                qubit_pauli_map=self._qubit_pauli_map, coefficient=self._coefficient * other
            )

        if isinstance(other, raw_types.Operation) and isinstance(other.gate, identity.IdentityGate):
            return self

        # Note: PauliString case handled by __mul__.
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Complex):
            return PauliString(
                qubit_pauli_map=self._qubit_pauli_map, coefficient=self._coefficient / other
            )
        return NotImplemented

    def __add__(self, other):
        return linear_combinations.PauliSum.from_pauli_strings(self).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return linear_combinations.PauliSum.from_pauli_strings(self).__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __contains__(self, key: TKey) -> bool:
        return key in self._qubit_pauli_map

    def _decompose_(self):
        if not self._has_unitary_():
            return None
        return [
            *(
                []
                if self.coefficient == 1
                else [global_phase_op.global_phase_operation(self.coefficient)]
            ),
            *[self[q].on(q) for q in self.qubits],
        ]

    def keys(self) -> KeysView[TKey]:
        """Returns the sequence of qubits on which this pauli string acts."""
        return self._qubit_pauli_map.keys()

    @property
    def qubits(self) -> Tuple[TKey, ...]:
        """Returns a tuple of qubits on which this pauli string acts."""
        return tuple(self.keys())

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> List[str]:
        if not len(self._qubit_pauli_map):
            return NotImplemented

        qs = args.known_qubits or list(self.keys())
        symbols = list(str(self.get(q)) for q in qs)
        if self.coefficient == 1:
            prefix = '+'
        elif self.coefficient == -1:
            prefix = '-'
        elif self.coefficient == 1j:
            prefix = 'i'
        elif self.coefficient == -1j:
            prefix = '-i'
        elif isinstance(self.coefficient, numbers.Number):
            prefix = f'({args.format_complex(self.coefficient)})*'
        else:
            prefix = f'({self.coefficient})*'
        symbols[0] = f'PauliString({prefix}{symbols[0]})'
        return symbols

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'PauliString':
        """Returns a new `PauliString` with `self.qubits` mapped to `new_qubits`.

        Args:
            new_qubits: The new qubits to replace `self.qubits` by.

        Returns:
            `PauliString` with mapped qubits.

        Raises:
            ValueError: If `len(new_qubits) != len(self.qubits)`.
        """
        if len(new_qubits) != len(self.qubits):
            raise ValueError(
                f'Number of new qubits: {len(new_qubits)} does not match '
                f'self.qubits: {len(self.qubits)}.'
            )
        return PauliString(
            qubit_pauli_map=dict(zip(new_qubits, (self[q] for q in self.qubits))),
            coefficient=self._coefficient,
        )

    def with_coefficient(self, new_coefficient: 'cirq.TParamValComplex') -> 'PauliString':
        """Returns a new `PauliString` with `self.coefficient` replaced with `new_coefficient`."""
        return PauliString(qubit_pauli_map=dict(self._qubit_pauli_map), coefficient=new_coefficient)

    def values(self) -> ValuesView[pauli_gates.Pauli]:
        """Ordered sequence of `cirq.Pauli` gates acting on `self.keys()`."""
        return self._qubit_pauli_map.values()

    def items(self) -> ItemsView[TKey, pauli_gates.Pauli]:
        """Returns (cirq.Qid, cirq.Pauli) pairs representing 1-qubit operations of pauli string."""
        return self._qubit_pauli_map.items()

    def frozen(self) -> 'cirq.PauliString':
        """Returns a `cirq.PauliString` with the same contents."""
        return self

    def mutable_copy(self) -> 'cirq.MutablePauliString':
        """Returns a new `cirq.MutablePauliString` with the same contents."""
        return MutablePauliString(
            coefficient=self.coefficient,
            pauli_int_dict={
                q: PAULI_GATE_LIKE_TO_INDEX_MAP[p] for q, p in self._qubit_pauli_map.items()
            },
        )

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._qubit_pauli_map.keys())

    def __bool__(self):
        return bool(self._qubit_pauli_map)

    def __len__(self) -> int:
        return len(self._qubit_pauli_map)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Print ASCII diagram in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('cirq.PauliString(...)')
        else:
            p.text(str(self))

    def __repr__(self) -> str:
        ordered_qubits = self.qubits
        prefix = ''

        factors = []
        if self._coefficient == -1:
            prefix = '-'
        else:
            factors.append(repr(self._coefficient))

        if not ordered_qubits:
            factors.append('cirq.PauliString()')
        for q in ordered_qubits:
            factors.append(repr(cast(raw_types.Gate, self[q]).on(q)))

        fused = prefix + '*'.join(factors)
        if len(factors) > 1:
            return f'({fused})'
        return fused

    def __str__(self) -> str:
        ordered_qubits = sorted(self.qubits)
        prefix = ''

        factors = []
        if self._coefficient == -1:
            prefix = '-'
        elif self._coefficient != 1:
            factors.append(repr(self._coefficient))

        if not ordered_qubits:
            factors.append('I')
        for q in ordered_qubits:
            factors.append(str(cast(raw_types.Gate, self[q]).on(q)))

        return prefix + '*'.join(factors)

    def matrix(self, qubits: Optional[Iterable[TKey]] = None) -> np.ndarray:
        """Returns the matrix of self in computational basis of qubits.

        Args:
            qubits: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the Pauli string is to
                be computed. Qubits absent from `self.qubits` are acted on by
                the identity. Defaults to `self.qubits`.

        Raises:
            NotImplementedError: If this PauliString is parameterized.
        """
        qubits = self.qubits if qubits is None else qubits
        factors = [self.get(q, default=identity.I) for q in qubits]
        if protocols.is_parameterized(self):
            raise NotImplementedError('Cannot express as matrix when parameterized')
        assert isinstance(self.coefficient, complex)
        return linalg.kron(self.coefficient, *[protocols.unitary(f) for f in factors])

    def _has_unitary_(self) -> bool:
        if self._is_parameterized_():
            return False
        return abs(1 - abs(cast(complex, self.coefficient))) < 1e-6

    def _unitary_(self) -> Optional[np.ndarray]:
        if not self._has_unitary_():
            return None
        return self.matrix()

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'):
        if not self._has_unitary_():
            return None
        assert isinstance(self.coefficient, numbers.Complex)
        if self.coefficient != 1:
            args.target_tensor *= self.coefficient
        return protocols.apply_unitaries([self[q].on(q) for q in self.qubits], self.qubits, args)

    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qubit_map: Mapping[TKey, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        r"""Evaluate the expectation of this PauliString given a state vector.

        Compute the expectation value of this PauliString with respect to a
        state vector. By convention expectation values are defined for Hermitian
        operators, and so this method will fail if this PauliString is
        non-Hermitian.

        `state` must be an array representation of a state vector and have
        shape `(2 ** n, )` or `(2, 2, ..., 2)` (n entries) where `state` is
        expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliString to the
                indices of the qubits that `state_vector` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state_vector` represents
                a valid state vector.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If this PauliString is non-Hermitian or
                parameterized.
            TypeError: If the input state is not complex.
            ValueError: If the input state does not have the correct shape.
        """
        if self._is_parameterized_():
            raise NotImplementedError('Cannot get expectation value when parameterized')

        # cast since Expressions will be forbidden by the above statement.
        if abs(cast(complex, self.coefficient).imag) > 0.0001:
            raise NotImplementedError(
                'Cannot compute expectation value of a non-Hermitian '
                f'PauliString <{self}>. Coefficient must be real.'
            )

        # FIXME: Avoid enforce specific complex type. This is necessary to
        # prevent an `apply_unitary` bug (Issue #2041).
        if state_vector.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or np.complex128")

        size = state_vector.size
        num_qubits = size.bit_length() - 1
        if len(state_vector.shape) != 1 and state_vector.shape != (2,) * num_qubits:
            raise ValueError(
                "Input array does not represent a state vector "
                "with shape `(2 ** n,)` or `(2, ..., 2)`."
            )

        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        if check_preconditions:
            qis.validate_normalized_state_vector(
                state_vector=state_vector,
                qid_shape=(2,) * num_qubits,
                dtype=state_vector.dtype,
                atol=atol,
            )
        return self._expectation_from_state_vector_no_validation(state_vector, qubit_map)

    def _expectation_from_state_vector_no_validation(
        self, state_vector: np.ndarray, qubit_map: Mapping[TKey, int]
    ) -> float:
        """Evaluate the expectation of this PauliString given a state vector.

        This method does not provide input validation. See
        `PauliString.expectation_from_state_vector` for function description.

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        if len(state_vector.shape) == 1:
            num_qubits = state_vector.shape[0].bit_length() - 1
            state_vector = np.reshape(state_vector, (2,) * num_qubits)

        ket = np.copy(state_vector)
        for qubit, pauli in self.items():
            buffer = np.empty(ket.shape, dtype=state_vector.dtype)
            args = protocols.ApplyUnitaryArgs(
                target_tensor=ket, available_buffer=buffer, axes=(qubit_map[qubit],)
            )
            ket = protocols.apply_unitary(pauli, args)

        return self.coefficient * (
            np.tensordot(state_vector.conj(), ket, axes=len(ket.shape)).item()
        )

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qubit_map: Mapping[TKey, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        r"""Evaluate the expectation of this PauliString given a density matrix.

        Compute the expectation value of this PauliString with respect to an
        array representing a density matrix. By convention expectation values
        are defined for Hermitian operators, and so this method will fail if
        this PauliString is non-Hermitian.

        `state` must be an array representation of a density matrix and have
        shape `(2 ** n, 2 ** n)` or `(2, 2, ..., 2)` (2*n entries), where
        `state` is expressed over n qubits.

        `qubit_map` must assign an integer index to each qubit in this
        PauliString that determines which bit position of a computational basis
        state that qubit corresponds to. For example if `state` represents
        $|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

            cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
            cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
                indices of the qubits that `state` is defined over.
            atol: Absolute numerical tolerance.
            check_preconditions: Whether to check that `state` represents a
                valid density matrix.

        Returns:
            The expectation value of the input state.

        Raises:
            NotImplementedError: If this PauliString is non-Hermitian or
                parameterized.
            TypeError: If the input state is not complex.
            ValueError: If the input state does not have the correct shape.
        """
        if self._is_parameterized_():
            raise NotImplementedError('Cannot get expectation value when parameterized')
        if abs(cast(complex, self.coefficient).imag) > 0.0001:
            raise NotImplementedError(
                'Cannot compute expectation value of a non-Hermitian '
                f'PauliString <{self}>. Coefficient must be real.'
            )

        # FIXME: Avoid enforcing specific complex type. This is necessary to
        # prevent an `apply_unitary` bug (Issue #2041).
        if state.dtype.kind != 'c':
            raise TypeError("Input state dtype must be np.complex64 or np.complex128")

        size = state.size
        num_qubits = int(np.sqrt(size)).bit_length() - 1
        dim = 1 << num_qubits
        if state.shape != (dim, dim) and state.shape != (2, 2) * num_qubits:
            raise ValueError(
                "Input array does not represent a density matrix "
                "with shape `(2 ** n, 2 ** n)` or `(2, ..., 2)`."
            )

        _validate_qubit_mapping(qubit_map, self.qubits, num_qubits)
        if check_preconditions:
            # Do not enforce reshaping if the state all axes are dimension 2.
            _ = qis.to_valid_density_matrix(
                density_matrix_rep=state.reshape(dim, dim),
                num_qubits=num_qubits,
                dtype=state.dtype,
                atol=atol,
            )
        return self._expectation_from_density_matrix_no_validation(state, qubit_map)

    def _expectation_from_density_matrix_no_validation(
        self, state: np.ndarray, qubit_map: Mapping[TKey, int]
    ) -> float:
        """Evaluate the expectation of this PauliString given a density matrix.

        This method does not provide input validation. See
        `PauliString.expectation_from_density_matrix` for function description.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this PauliString to the
            indices of the qubits that `state` is defined over.

        Returns:
            The expectation value of the input state.
        """
        result = np.copy(state)
        if len(state.shape) == 2:
            num_qubits = state.shape[0].bit_length() - 1
            result = np.reshape(result, (2,) * num_qubits * 2)

        for qubit, pauli in self.items():
            buffer = np.empty(result.shape, dtype=state.dtype)
            args = protocols.ApplyUnitaryArgs(
                target_tensor=result, available_buffer=buffer, axes=(qubit_map[qubit],)
            )
            result = protocols.apply_unitary(pauli, args)

        while any(result.shape):
            result = np.trace(result, axis1=0, axis2=len(result.shape) // 2)

        return float(np.real(result * self.coefficient))

    def zip_items(
        self, other: 'cirq.PauliString[TKey]'
    ) -> Iterator[Tuple[TKey, Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]]:
        """Combines pauli operations from pauli strings in a qubit-by-qubit fashion.

        For every qubit that has a `cirq.Pauli` operation acting on it in both `self` and `other`,
        the method yields a tuple corresponding to `(qubit, (pauli_in_self, pauli_in_other))`.

        Args:
            other: The other `cirq.PauliString` to zip pauli operations with.

        Returns:
            A sequence of `(qubit, (pauli_in_self, pauli_in_other))` tuples for every `qubit`
            that has a `cirq.Pauli` operation acting on it in both `self` and `other.
        """
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield qubit, (pauli0, other[qubit])

    def zip_paulis(
        self, other: 'cirq.PauliString'
    ) -> Iterator[Tuple[pauli_gates.Pauli, pauli_gates.Pauli]]:
        """Combines pauli operations from pauli strings in a qubit-by-qubit fashion.

        For every qubit that has a `cirq.Pauli` operation acting on it in both `self` and `other`,
        the method yields a tuple corresponding to `(pauli_in_self, pauli_in_other)`.

        Args:
            other: The other `cirq.PauliString` to zip pauli operations with.

        Returns:
            A sequence of `(pauli_in_self, pauli_in_other)` tuples for every `qubit`
            that has a `cirq.Pauli` operation acting on it in both `self` and `other.
        """
        return (paulis for qubit, paulis in self.zip_items(other))

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        if not isinstance(other, PauliString):
            return NotImplemented
        return sum(not protocols.commutes(p0, p1) for p0, p1 in self.zip_paulis(other)) % 2 == 0

    def __neg__(self) -> 'PauliString':
        return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=-self._coefficient)

    def __pos__(self) -> 'PauliString':
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override numpy behavior."""
        if ufunc == np.exp and len(inputs) == 1 and inputs[0] is self:
            return math.e**self
        if ufunc == np.multiply and len(inputs) == 2 and inputs[1] is self:
            return self * inputs[0]
        return NotImplemented

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PauliString(
                qubit_pauli_map=self._qubit_pauli_map, coefficient=self.coefficient**-1
            )
        if self._is_parameterized_():
            return NotImplemented
        if isinstance(power, (int, float)):
            r, i = cmath.polar(self.coefficient)
            if abs(r - 1) > 0.0001:
                # Raising non-unitary PauliStrings to a power is not supported.
                return NotImplemented

            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {
                    pauli_gates.X: common_gates.XPowGate,
                    pauli_gates.Y: common_gates.YPowGate,
                    pauli_gates.Z: common_gates.ZPowGate,
                }
                return gates[p](exponent=power).on(q)

            global_half_turns = power * (i / math.pi)

            # HACK: Avoid circular dependency.
            from cirq.ops import pauli_string_phasor

            return pauli_string_phasor.PauliStringPhasor(
                PauliString(qubit_pauli_map=self._qubit_pauli_map),
                exponent_neg=global_half_turns + power,
                exponent_pos=global_half_turns,
            )
        return NotImplemented

    def __rpow__(self, base):
        if self._is_parameterized_():
            return NotImplemented
        if isinstance(base, (int, float)) and base > 0:
            if abs(self.coefficient.real) > 0.0001:
                raise NotImplementedError(
                    'Exponentiated to a non-Hermitian PauliString '
                    f'<{base}**{self}>. Coefficient must be imaginary.'
                )

            half_turns = 2 * math.log(base) * (-self.coefficient.imag / math.pi)

            if len(self) == 1:
                q, p = next(iter(self.items()))
                gates = {
                    pauli_gates.X: common_gates.XPowGate,
                    pauli_gates.Y: common_gates.YPowGate,
                    pauli_gates.Z: common_gates.ZPowGate,
                }
                return gates[p](exponent=half_turns, global_shift=-0.5).on(q)

            # HACK: Avoid circular dependency.
            from cirq.ops import pauli_string_phasor

            return pauli_string_phasor.PauliStringPhasor(
                PauliString(qubit_pauli_map=self._qubit_pauli_map),
                exponent_neg=+half_turns / 2,
                exponent_pos=-half_turns / 2,
            )
        return NotImplemented

    def map_qubits(self, qubit_map: Dict[TKey, TKeyNew]) -> 'cirq.PauliString[TKeyNew]':
        """Replaces every qubit `q` in `self.qubits` with `qubit_map[q]`.

        Args:
            qubit_map: A map from qubits in the pauli string to new qubits.

        Returns:
            A new `PauliString` with remapped qubits.

        Raises:
            ValueError: If the map does not contain an entry for all qubits in the pauli string.
        """
        if not set(self.qubits) <= qubit_map.keys():
            raise ValueError(
                "qubit_map must have a key for every qubit in the pauli strings' qubits. "
                f"keys: {qubit_map.keys()} pauli string qubits: {self.qubits}"
            )
        new_qubit_pauli_map = {qubit_map[qubit]: pauli for qubit, pauli in self.items()}
        return PauliString(qubit_pauli_map=new_qubit_pauli_map, coefficient=self._coefficient)

    def to_z_basis_ops(self) -> Iterator[raw_types.Operation]:
        """Returns single qubit operations to convert the qubits to the computational basis."""
        for qubit, pauli in self.items():
            yield clifford_gate.SingleQubitCliffordGate.from_single_map(
                {pauli: (pauli_gates.Z, False)}
            )(qubit)

    def dense(self, qubits: Sequence[TKey]) -> 'cirq.DensePauliString':
        """Returns a `cirq.DensePauliString` version of this Pauli string.

        This method satisfies the invariant `P.dense(qubits).on(*qubits) == P`.

        Args:
            qubits: The implicit sequence of qubits used by the dense pauli
                string. Specifically, if the returned dense Pauli string was
                applied to these qubits (via its `on` method) then the result
                would be a Pauli string equivalent to the receiving Pauli
                string.

        Returns:
            A `cirq.DensePauliString` instance `D` such that `D.on(*qubits)`
            equals the receiving `cirq.PauliString` instance `P`.

        Raises:
            ValueError: If the number of qubits is too small.
        """

        if not self.keys() <= set(qubits):
            raise ValueError('not self.keys() <= set(qubits)')
        # pylint: disable=too-many-function-args
        pauli_mask = [self.get(q, identity.I) for q in qubits]
        # pylint: enable=too-many-function-args
        return dense_pauli_string.DensePauliString(pauli_mask, coefficient=self.coefficient)

    def conjugated_by(self, clifford: 'cirq.OP_TREE') -> 'PauliString':
        r"""Returns the Pauli string conjugated by a clifford operation.

        The product-of-Paulis $P$ conjugated by the Clifford operation $C$ is

            $$
            C^\dagger P C
            $$

        For example, conjugating a +Y operation by an S operation results in a
        +X operation (as opposed to a -X operation).

        In a circuit diagram where `P` is a pauli string observable immediately
        after a Clifford operation `C`, the pauli string `P.conjugated_by(C)` is
        the equivalent pauli string observable just before `C`.

            --------------------------C---P---

            = ---C---P------------------------

            = ---C---P---------C^-1---C-------

            = ---C---P---C^-1---------C-------

            = --(C^-1 · P · C)--------C-------

            = ---P.conjugated_by(C)---C-------

        Analogously, a Pauli product P can be moved from before a Clifford C in
        a circuit diagram to after the Clifford C by conjugating P by the
        inverse of C:

            ---P---C---------------------------

            = -----C---P.conjugated_by(C^-1)---

        Args:
            clifford: The Clifford operation to conjugate by. This can be an
                individual operation, or a tree of operations.

                Note that the composite Clifford operation defined by a sequence
                of operations is equivalent to a circuit containing those
                operations in the given order. Somewhat counter-intuitively,
                this means that the operations in the sequence are conjugated
                onto the Pauli string in reverse order. For example,
                `P.conjugated_by([C1, C2])` is equivalent to
                `P.conjugated_by(C2).conjugated_by(C1)`.

        Examples:
            >>> a, b = cirq.LineQubit.range(2)
            >>> print(cirq.X(a).conjugated_by(cirq.CZ(a, b)))
            X(q(0))*Z(q(1))
            >>> print(cirq.X(a).conjugated_by(cirq.S(a)))
            -Y(q(0))
            >>> print(cirq.X(a).conjugated_by([cirq.H(a), cirq.CNOT(a, b)]))
            Z(q(0))*X(q(1))

        Returns:
            The Pauli string conjugated by the given Clifford operation.
        """

        # Initialize the ps the same as self.
        ps = PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=self.coefficient)
        all_ops = list(op_tree.flatten_to_ops(clifford))
        all_qubits = set.union(set(self.qubits), [q for op in all_ops for q in op.qubits])
        # Iteratively calculate the conjugation in reverse order of ops.
        for op in all_ops[::-1]:
            # To calcuate the conjugation of P (`ps`) with respect to C (`op`)
            # Decompose P = Pc⊗R, where Pc acts on the same qubits as C, R acts on the remaining.
            # Then the conjugation = (C^{-1}⊗I·Pc⊗R·C⊗I) = (C^{-1}·Pc·C)⊗R.

            # Isolate R
            remain: 'cirq.PauliString' = PauliString(
                *(pauli(q) for q in all_qubits - set(op.qubits) if (pauli := ps.get(q)) is not None)
            )

            # Initialize the conjugation of Pc.
            conjugated: 'cirq.DensePauliString' = (
                dense_pauli_string.DensePauliString(pauli_mask=[identity.I for _ in op.qubits])
                * ps.coefficient
            )

            # Calculate the conjugation via CliffordGate's clifford_tableau.
            # Note the clifford_tableau in CliffordGate represents C·P·C^-1 instead of C^-1·P·C.
            # So we take the inverse of the tableau to match the definition of the conjugation here.
            gate_in_clifford: 'cirq.CliffordGate'
            if isinstance(op.gate, clifford_gate.CliffordGate):
                gate_in_clifford = op.gate
            else:
                # Convert the clifford gate to CliffordGate type.
                gate_in_clifford = clifford_gate.CliffordGate.from_op_list([op], op.qubits)
            tableau = gate_in_clifford.clifford_tableau.inverse()

            # Calculate the conjugation by `op` via mutiplying the conjugation of each Pauli:
            #   C^{-1}·(P_1⊗...⊗P_n)·C
            # = C^{-1}·(P_1⊗I) ...·(P_n⊗I)·C
            # = (C^{-1}(P_1⊗I)C)·...·(C^{-1}(P_n⊗I)C)
            # For the Pauli on the kth qubit P_k. The conjugation is calculated as following.
            #   Puali X_k's conjugation is from the destabilzer table;
            #   Puali Z_k's conjugation is from the stabilzer table;
            #   Puali Y_k's conjugation is calcluated according to Y = iXZ. E.g., for the kth qubit,
            #     C^{-1}·Y_k⊗I·C = C^{-1}·(iX_k⊗I·Z_k⊗I)·C = i (C^{-1}·X_k⊗I·C)·(C^{-1}·Z_k⊗I·C).
            for qid, qubit in enumerate(op.qubits):
                pauli = ps.get(qubit)
                match pauli:
                    case None:
                        continue
                    case pauli_gates.X:
                        conjugated *= tableau.destabilizers()[qid]
                    case pauli_gates.Z:
                        conjugated *= tableau.stabilizers()[qid]
                    case pauli_gates.Y:
                        conjugated *= (
                            1j
                            * tableau.destabilizers()[qid]  # conj X first
                            * tableau.stabilizers()[qid]  # then conj Z
                        )
            ps = remain * conjugated.on(*op.qubits)
        return ps

    def after(self, ops: 'cirq.OP_TREE') -> 'cirq.PauliString':
        r"""Determines the equivalent pauli string after some operations.

        If the PauliString is $P$ and the Clifford operation is $C$, then the
        result is $C P C^\dagger$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The result of propagating this pauli string from before to after the
            given operations.
        """
        return self.conjugated_by(protocols.inverse(ops))

    def before(self, ops: 'cirq.OP_TREE') -> 'cirq.PauliString':
        r"""Determines the equivalent pauli string before some operations.

        If the PauliString is $P$ and the Clifford operation is $C$, then the
        result is $C^\dagger P C$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The result of propagating this pauli string from after to before the
            given operations.
        """
        return self.conjugated_by(ops)

    def pass_operations_over(
        self, ops: Iterable['cirq.Operation'], after_to_before: bool = False
    ) -> 'PauliString':
        """Determines how the Pauli string changes when conjugated by Cliffords.

        The output and input pauli strings are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_STRING───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_STRING───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
        # TODO(#6946): deprecate this method.
        # Note: This method is supposed to be replaced by conjugated_by()
        #  (see #2351 for details).
        if after_to_before:
            return self.after(ops)

        if isinstance(ops, gate_operation.GateOperation):
            return self.before(ops)

        all_ops = list(op_tree.flatten_to_ops(ops))
        return self.before(all_ops[::-1])

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.coefficient)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.coefficient)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'cirq.PauliString':
        coefficient = protocols.resolve_parameters(self.coefficient, resolver, recursive)
        return PauliString(qubit_pauli_map=self._qubit_pauli_map, coefficient=coefficient)


def _validate_qubit_mapping(
    qubit_map: Mapping[TKey, int], pauli_qubits: Tuple[TKey, ...], num_state_qubits: int
) -> None:
    """Validates that a qubit map is a valid mapping.

    This will enforce that all elements of `pauli_qubits` appear in `qubit_map`,
    and that the integers in `qubit_map` correspond to valid positions in a
    representation of a state over `num_state_qubits`.

    Args:
        qubit_map: A map from qubits to integers.
        pauli_qubits: The qubits that must be contained in `qubit_map`.
        num_state_qubits: The number of qubits over which a state is expressed.

    Raises:
        TypeError: If the qubit map is between the wrong types.
        ValueError: If the qubit maps is not complete or does not match with
            `num_state_qubits`.
    """
    if not isinstance(qubit_map, Mapping) or not all(
        isinstance(k, raw_types.Qid) and isinstance(v, int) for k, v in qubit_map.items()
    ):
        raise TypeError(
            "Input qubit map must be a valid mapping from Qubit ID's to integer indices."
        )

    if not set(qubit_map.keys()) >= set(pauli_qubits):
        raise ValueError(
            "Input qubit map must be a complete mapping over all of this PauliString's qubits."
        )

    used_inds = [qubit_map[q] for q in pauli_qubits]
    if len(used_inds) != len(set(used_inds)) or not set(range(num_state_qubits)) >= set(
        sorted(used_inds)
    ):
        raise ValueError(
            f'Input qubit map indices must be valid for a state over {num_state_qubits} qubits.'
        )


def _try_interpret_as_pauli_string(op: Any):
    """Return a reprepresentation of an operation as a pauli string, if it is possible."""
    if isinstance(op, gate_operation.GateOperation):
        gates = {
            common_gates.XPowGate: pauli_gates.X,
            common_gates.YPowGate: pauli_gates.Y,
            common_gates.ZPowGate: pauli_gates.Z,
        }
        if (pauli := gates.get(type(op.gate), None)) is not None:
            exponent = op.gate.exponent  # type: ignore
            if exponent % 2 == 0:
                return PauliString()
            if exponent % 2 == 1:
                return pauli.on(op.qubits[0])
    return None


# Ignoring type because mypy believes `with_qubits` methods are incompatible.
class SingleQubitPauliStringGateOperation(  # type: ignore
    gate_operation.GateOperation, PauliString
):
    """An operation to represent single qubit pauli gates applied to a qubit.

    Satisfies the contract of both `cirq.GateOperation` and `cirq.PauliString`. Relies
    implicitly on the fact that PauliString({q: X}) compares as equal to
    GateOperation(X, [q]).
    """

    def __init__(self, pauli: pauli_gates.Pauli, qubit: 'cirq.Qid'):
        PauliString.__init__(self, qubit_pauli_map={qubit: pauli})
        gate_operation.GateOperation.__init__(self, cast(raw_types.Gate, pauli), [qubit])

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'SingleQubitPauliStringGateOperation':
        if len(new_qubits) != 1:
            raise ValueError("len(new_qubits) != 1")
        return SingleQubitPauliStringGateOperation(
            cast(pauli_gates.Pauli, self.gate), new_qubits[0]
        )

    @property
    def pauli(self) -> pauli_gates.Pauli:
        return cast(pauli_gates.Pauli, self.gate)

    @property
    def qubit(self) -> raw_types.Qid:
        assert len(self.qubits) == 1
        return self.qubits[0]

    def _as_pauli_string(self) -> PauliString:
        return PauliString(qubit_pauli_map={self.qubit: self.pauli})

    def __mul__(self, other):
        if isinstance(other, SingleQubitPauliStringGateOperation):
            return self._as_pauli_string() * other._as_pauli_string()
        if isinstance(other, (PauliString, numbers.Complex)):
            return self._as_pauli_string() * other
        if (as_pauli_string := _try_interpret_as_pauli_string(other)) is not None:
            return self * as_pauli_string
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (PauliString, numbers.Complex)):
            return other * self._as_pauli_string()
        if (as_pauli_string := _try_interpret_as_pauli_string(other)) is not None:
            return as_pauli_string * self
        return NotImplemented

    def __neg__(self):
        return -self._as_pauli_string()

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['pauli', 'qubit'])

    @classmethod
    def _from_json_dict_(  # type: ignore
        cls, pauli: pauli_gates.Pauli, qubit: 'cirq.Qid', **kwargs
    ):
        # Note, this method is required or else superclasses' deserialization
        # would be used
        return cls(pauli=pauli, qubit=qubit)


@value.value_equality(unhashable=True, manual_cls=True, approximate=True)
class MutablePauliString(Generic[TKey]):
    """Mutable version of `cirq.PauliString`, used mainly for efficiently mutating pauli strings.

    `cirq.MutablePauliString` is a mutable version of `cirq.PauliString`, which is often
    useful for mutating pauli strings efficiently instead of always creating a copy. Note
    that, unlike `cirq.PauliString`, `MutablePauliString` is not a `cirq.Operation`.

    It exists mainly to help mutate pauli strings efficiently and then convert back to a
    frozen `cirq.PauliString` representation, which can then be used as operators or
    observables.
    """

    def __init__(
        self,
        *contents: 'cirq.PAULI_STRING_LIKE',
        coefficient: 'cirq.TParamValComplex' = 1,
        pauli_int_dict: Optional[Dict[TKey, int]] = None,
    ):
        """Initializes a new `MutablePauliString`.

        Args:
            *contents: A value or values to convert into a pauli string. This
                can be a number, a pauli operation, a dictionary from qubit to
                pauli/identity gates, or collections thereof. If a list of
                values is given, they are each individually converted and then
                multiplied from left to right in order.
            coefficient: Initial scalar coefficient or symbol. Defaults to 1.
            pauli_int_dict: Initial dictionary mapping qubits to integers corresponding
                to pauli operations. Defaults to the empty dictionary. Note that, unlike
                dictionaries passed to contents, this dictionary must not contain values
                corresponding to identity gates; i.e. all integer values must be between
                [1, 3]. Further note that this argument specifies values that are logically
                *before* factors specified in `contents`; `contents` are *right* multiplied
                onto the values in this dictionary.

        Raises:
            ValueError: If the `pauli_int_dict` has integer values `v` not satisfying `1 <= v <= 3`.
        """
        self.coefficient: Union[sympy.Expr, 'cirq.TParamValComplex'] = (
            coefficient if isinstance(coefficient, sympy.Expr) else complex(coefficient)
        )
        if pauli_int_dict is not None:
            for v in pauli_int_dict.values():
                if not 1 <= v <= 3:
                    raise ValueError(f"Value {v} of pauli_int_dict must be between 1 and 3.")
        self.pauli_int_dict: Dict[TKey, int] = {} if pauli_int_dict is None else pauli_int_dict
        if contents:
            self.inplace_left_multiply_by(contents)

    def _value_equality_values_(self):
        return self.frozen()._value_equality_values_()

    def _value_equality_values_cls_(self):
        return self.frozen()._value_equality_values_cls_()

    def _imul_atom_helper(self, key: TKey, pauli_lhs: int, sign: int) -> int:
        pauli_old = self.pauli_int_dict.pop(key, 0)
        pauli_new = pauli_lhs ^ pauli_old
        if pauli_new:
            self.pauli_int_dict[key] = pauli_new
        if not pauli_lhs or not pauli_old or pauli_lhs == pauli_old:
            return 0
        if (pauli_old - pauli_lhs) % 3 == 1:
            return sign
        return -sign

    def keys(self) -> AbstractSet[TKey]:
        """Returns the sequence of qubits on which this pauli string acts."""
        return self.pauli_int_dict.keys()

    def values(self) -> Iterator['cirq.Pauli']:
        """Ordered sequence of `cirq.Pauli` gates acting on `self.keys()`."""
        for v in self.pauli_int_dict.values():
            yield _INT_TO_PAULI[v - 1]

    def __iter__(self) -> Iterator[TKey]:
        return iter(self.pauli_int_dict)

    def __len__(self) -> int:
        return len(self.pauli_int_dict)

    def __bool__(self) -> bool:
        return bool(self.pauli_int_dict)

    def frozen(self) -> 'cirq.PauliString':
        """Returns a `cirq.PauliString` with the same contents.

        For example, this is useful because `cirq.PauliString` is an operation
        whereas `cirq.MutablePauliString` is not.
        """
        return PauliString(
            coefficient=self.coefficient,
            qubit_pauli_map={q: _INT_TO_PAULI[p - 1] for q, p in self.pauli_int_dict.items() if p},
        )

    def mutable_copy(self) -> 'cirq.MutablePauliString':
        """Returns a new `cirq.MutablePauliString` with the same contents."""
        return MutablePauliString(
            coefficient=self.coefficient, pauli_int_dict=dict(self.pauli_int_dict)
        )

    def items(self) -> Iterator[Tuple[TKey, 'cirq.Pauli']]:
        """Returns (cirq.Qid, cirq.Pauli) pairs representing 1-qubit operations of pauli string."""
        for k, v in self.pauli_int_dict.items():
            yield k, _INT_TO_PAULI[v - 1]

    def __contains__(self, item: Any) -> bool:
        return item in self.pauli_int_dict

    def __getitem__(self, item: Any) -> 'cirq.Pauli':
        return _INT_TO_PAULI[self.pauli_int_dict[item] - 1]

    def __setitem__(self, key: TKey, value: 'cirq.PAULI_GATE_LIKE'):
        value = _pauli_like_to_pauli_int(key, value)
        if value:
            self.pauli_int_dict[key] = _pauli_like_to_pauli_int(key, value)
        else:
            self.pauli_int_dict.pop(key, None)

    def __delitem__(self, key: TKey):
        del self.pauli_int_dict[key]

    # pylint: disable=function-redefined
    @overload
    def get(self, key: TKey, default: None = None) -> Union['cirq.Pauli', None]:
        pass

    @overload
    def get(self, key: TKey, default: TDefault) -> Union['cirq.Pauli', TDefault]:
        pass

    def get(self, key: TKey, default=None) -> Union['cirq.Pauli', TDefault, None]:
        """Returns the `cirq.Pauli` operation acting on qubit `key` or `default` if none exists."""
        result = self.pauli_int_dict.get(key, None)
        return default if result is None else _INT_TO_PAULI[result - 1]

    # pylint: enable=function-redefined
    def inplace_before(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
        r"""Propagates the pauli string from after to before a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C^\dagger P C$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.
        """
        return self.inplace_after(protocols.inverse(ops))

    def inplace_after(self, ops: 'cirq.OP_TREE') -> 'cirq.MutablePauliString':
        r"""Propagates the pauli string from before to after a Clifford effect.

        If the old value of the MutablePauliString is $P$ and the Clifford
        operation is $C$, then the new value of the MutablePauliString is
        $C P C^\dagger$.

        Args:
            ops: A stabilizer operation or nested collection of stabilizer
                operations.

        Returns:
            The mutable pauli string that was mutated.

        Raises:
            NotImplementedError: If any ops decompose into an unsupported
                Clifford gate.
        """
        for clifford in op_tree.flatten_to_ops(ops):
            for op in _decompose_into_cliffords(clifford):
                ps = [self.pauli_int_dict.pop(cast(TKey, q), 0) for q in op.qubits]
                if not any(ps):
                    continue
                gate = op.gate

                if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
                    out = gate.pauli_tuple(_INT_TO_PAULI[ps[0] - 1])
                    if out[1]:
                        self.coefficient *= -1
                    self.pauli_int_dict[cast(TKey, op.qubits[0])] = PAULI_GATE_LIKE_TO_INDEX_MAP[
                        out[0]
                    ]

                elif isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
                    q0, q1 = op.qubits
                    p0 = _INT_TO_PAULI_OR_IDENTITY[ps[0]]
                    p1 = _INT_TO_PAULI_OR_IDENTITY[ps[1]]

                    # Kick across Paulis that anti-commute with the controls.
                    kickback_0_to_1 = not protocols.commutes(p0, gate.pauli0)
                    kickback_1_to_0 = not protocols.commutes(p1, gate.pauli1)
                    kick0 = gate.pauli1 if kickback_0_to_1 else identity.I
                    kick1 = gate.pauli0 if kickback_1_to_0 else identity.I
                    self.__imul__({q0: p0, q1: kick0})
                    self.__imul__({q0: kick1, q1: p1})

                    # Decompose inverted controls into single-qubit operations.
                    if gate.invert0:
                        self.inplace_after(gate.pauli1(q1))
                    if gate.invert1:
                        self.inplace_after(gate.pauli0(q0))

                else:  # pragma: no cover
                    raise NotImplementedError(f"Unrecognized decomposed Clifford: {op!r}")
        return self

    def _imul_helper(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
        """Left-multiplies or right-multiplies by a PAULI_STRING_LIKE.

        Args:
            other: What to multiply by.
            sign: +1 to left-multiply, -1 to right-multiply.

        Returns:
            self on success, NotImplemented given an unknown type of value.
        """
        if isinstance(other, (Mapping, PauliString, MutablePauliString)):
            if isinstance(other, (PauliString, MutablePauliString)):
                self.coefficient *= other.coefficient
            phase_log_i = 0
            for qubit, pauli_gate_like in other.items():
                pauli_int = _pauli_like_to_pauli_int(qubit, pauli_gate_like)
                phase_log_i += self._imul_atom_helper(cast(TKey, qubit), pauli_int, sign)
            self.coefficient *= 1j ** (phase_log_i & 3)
        elif isinstance(other, numbers.Complex):
            self.coefficient *= other
        elif isinstance(other, raw_types.Operation) and isinstance(
            other.gate, identity.IdentityGate
        ):
            pass
        elif (
            isinstance(other, Iterable)
            and not isinstance(other, str)
            and not isinstance(other, linear_combinations.PauliSum)
        ):
            if sign == +1:
                other = iter(reversed(list(other)))
            for item in other:
                if self._imul_helper(cast(PAULI_STRING_LIKE, item), sign) is NotImplemented:
                    return NotImplemented
        else:
            return NotImplemented

        return self

    def _imul_helper_checkpoint(self, other: 'cirq.PAULI_STRING_LIKE', sign: int):
        """Like `_imul_helper` but guarantees no-op on error."""

        if not isinstance(other, (numbers.Number, PauliString, MutablePauliString)):
            other = MutablePauliString()._imul_helper(other, sign)
            if other is NotImplemented:
                return NotImplemented
        return self._imul_helper(other, sign)

    def inplace_left_multiply_by(
        self, other: 'cirq.PAULI_STRING_LIKE'
    ) -> 'cirq.MutablePauliString':
        """Left-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to left-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was mutated.

        Raises:
            TypeError: `other` was not a `cirq.PAULI_STRING_LIKE`. `self`
                was not mutated.
        """
        if self._imul_helper_checkpoint(other, -1) is NotImplemented:
            raise TypeError(f"{other!r} is not cirq.PAULI_STRING_LIKE.")
        return self

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            # JSON requires mappings to have string keys.
            'pauli_int_dict': list(self.pauli_int_dict.items()),
            'coefficient': self.coefficient,
        }

    @classmethod
    def _from_json_dict_(cls, pauli_int_dict, coefficient, **kwargs):
        return cls(pauli_int_dict=dict(pauli_int_dict), coefficient=coefficient)

    def inplace_right_multiply_by(
        self, other: 'cirq.PAULI_STRING_LIKE'
    ) -> 'cirq.MutablePauliString':
        """Right-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to right-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was mutated.

        Raises:
            TypeError: `other` was not a `cirq.PAULI_STRING_LIKE`. `self`
                was not mutated.
        """
        if self._imul_helper_checkpoint(other, +1) is NotImplemented:
            raise TypeError(f"{other!r} is not cirq.PAULI_STRING_LIKE.")
        return self

    def __neg__(self) -> 'cirq.MutablePauliString':
        result = self.mutable_copy()
        result.coefficient *= -1
        return result

    def __pos__(self) -> 'cirq.MutablePauliString':
        return self.mutable_copy()

    def transform_qubits(
        self, func: Callable[[TKey], TKeyNew], *, inplace: bool = False
    ) -> 'cirq.MutablePauliString[TKeyNew]':
        """Returns a `MutablePauliString` with transformed qubits.

        Args:
            func: The qubit transformation to apply.
            inplace: If false (the default), creates a new mutable pauli string
                to store the result. If true, overwrites this mutable pauli
                string's contents. Defaults to false for consistency with
                `cirq.PauliString.transform_qubits` in situations where the
                pauli string being used may or may not be mutable.

        Returns:
            A transformed MutablePauliString.
            If inplace=True, returns `self`.
            If inplace=False, returns a new instance.
        """
        new_dict = {func(q): p for q, p in self.pauli_int_dict.items()}
        if not inplace:
            return MutablePauliString(coefficient=self.coefficient, pauli_int_dict=new_dict)
        result = cast('cirq.MutablePauliString[TKeyNew]', self)
        result.pauli_int_dict = new_dict
        return result

    def __imul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.MutablePauliString':
        """Left-multiplies a pauli string into this pauli string.

        Args:
            other: A pauli string or `cirq.PAULI_STRING_LIKE` to left-multiply
                into `self`.

        Returns:
            The `self` mutable pauli string that was successfully mutated.

            If `other` is not a `cirq.PAULI_STRING_LIKE`, `self` is not mutated
            and `NotImplemented` is returned.

        """
        return self._imul_helper_checkpoint(other, +1)

    def __mul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.PauliString':
        """Multiplies two pauli-string-likes together.

        The result is not mutable.
        """
        return self.frozen() * other

    def __rmul__(self, other: 'cirq.PAULI_STRING_LIKE') -> 'cirq.PauliString':
        """Multiplies two pauli-string-likes together.

        The result is not mutable.
        """
        return other * self.frozen()

    def __str__(self) -> str:
        return f'mutable {self.frozen()}'

    def __repr__(self) -> str:
        return f'{self.frozen()!r}.mutable_copy()'


def _decompose_into_cliffords(op: 'cirq.Operation') -> List['cirq.Operation']:
    # An operation that can be ignored?
    if isinstance(op.gate, global_phase_op.GlobalPhaseGate):
        return []

    # Already a known Clifford?
    if isinstance(
        op.gate,
        (clifford_gate.SingleQubitCliffordGate, pauli_interaction_gate.PauliInteractionGate),
    ):
        return [op]

    # Specifies a decomposition into Cliffords?
    v = getattr(op, '_decompose_into_clifford_', None)
    if v is not None:
        result = v()
        if result is not None and result is not NotImplemented:
            return list(op_tree.flatten_to_ops(result))

    # Specifies a decomposition that happens to contain only Cliffords?
    decomposed = protocols.decompose_once(op, None)
    if decomposed is not None:
        return [out for sub_op in decomposed for out in _decompose_into_cliffords(sub_op)]

    raise TypeError(
        f'Operation is not a known Clifford and did not decompose into known Cliffords: {op!r}'
    )


def _pass_operation_over(
    pauli_map: Dict[TKey, pauli_gates.Pauli], op: 'cirq.Operation', after_to_before: bool = False
) -> bool:
    if isinstance(op, gate_operation.GateOperation):
        gate = op.gate
        if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
            return _pass_single_clifford_gate_over(
                pauli_map, gate, cast(TKey, op.qubits[0]), after_to_before=after_to_before
            )
        if isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
            return _pass_pauli_interaction_gate_over(
                pauli_map,
                gate,
                cast(TKey, op.qubits[0]),
                cast(TKey, op.qubits[1]),
                after_to_before=after_to_before,
            )
    raise NotImplementedError(f'Unsupported operation: {op!r}')


def _pass_single_clifford_gate_over(
    pauli_map: Dict[TKey, pauli_gates.Pauli],
    gate: clifford_gate.SingleQubitCliffordGate,
    qubit: TKey,
    after_to_before: bool = False,
) -> bool:
    if qubit not in pauli_map:
        return False  # pragma: no cover
    if not after_to_before:
        gate **= -1
    pauli, inv = gate.pauli_tuple(pauli_map[qubit])
    pauli_map[qubit] = pauli
    return inv


def _pass_pauli_interaction_gate_over(
    pauli_map: Dict[TKey, pauli_gates.Pauli],
    gate: pauli_interaction_gate.PauliInteractionGate,
    qubit0: TKey,
    qubit1: TKey,
    after_to_before: bool = False,
) -> bool:
    def merge_and_kickback(
        qubit: TKey,
        pauli_left: Optional[pauli_gates.Pauli],
        pauli_right: Optional[pauli_gates.Pauli],
        inv: bool,
    ) -> int:
        assert pauli_left is not None or pauli_right is not None
        if pauli_left is None or pauli_right is None:
            pauli_map[qubit] = cast(pauli_gates.Pauli, pauli_left or pauli_right)
            return 0
        if pauli_left == pauli_right:
            del pauli_map[qubit]
            return 0

        pauli_map[qubit] = pauli_left.third(pauli_right)
        if (pauli_left < pauli_right) ^ after_to_before:
            return int(inv) * 2 + 1

        return int(inv) * 2 - 1

    quarter_kickback = 0
    if qubit0 in pauli_map and not protocols.commutes(pauli_map[qubit0], gate.pauli0):
        quarter_kickback += merge_and_kickback(
            qubit1, gate.pauli1, pauli_map.get(qubit1), gate.invert1
        )
    if qubit1 in pauli_map and not protocols.commutes(pauli_map[qubit1], gate.pauli1):
        quarter_kickback += merge_and_kickback(
            qubit0, pauli_map.get(qubit0), gate.pauli0, gate.invert0
        )
    assert (
        quarter_kickback % 2 == 0
    ), 'Impossible condition.  quarter_kickback is either incremented twice or never.'
    return quarter_kickback % 4 == 2


# Mypy has extreme difficulty with these constants for some reason.
_i = cast(identity.IdentityGate, identity.I)  # type: ignore
_x = cast(pauli_gates.Pauli, pauli_gates.X)  # type: ignore
_y = cast(pauli_gates.Pauli, pauli_gates.Y)  # type: ignore
_z = cast(pauli_gates.Pauli, pauli_gates.Z)  # type: ignore

PAULI_GATE_LIKE_TO_INDEX_MAP: Dict['cirq.PAULI_GATE_LIKE', int] = {
    _i: 0,
    _x: 1,
    _y: 2,
    _z: 3,
    'I': 0,
    'X': 1,
    'Y': 2,
    'Z': 3,
    'i': 0,
    'x': 1,
    'y': 2,
    'z': 3,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

_INT_TO_PAULI_OR_IDENTITY: List[Union['cirq.Pauli', 'cirq.IdentityGate']] = [_i, _x, _y, _z]
_INT_TO_PAULI: List['cirq.Pauli'] = [_x, _y, _z]


PAULI_GATE_LIKE_TO_GATE_MAP: Dict[
    'cirq.PAULI_GATE_LIKE', Union['cirq.Pauli', 'cirq.IdentityGate']
] = {k: _INT_TO_PAULI_OR_IDENTITY[v] for k, v in PAULI_GATE_LIKE_TO_INDEX_MAP.items()}


def _pauli_like_to_pauli_int(key: Any, pauli_gate_like: PAULI_GATE_LIKE):
    pauli_int = PAULI_GATE_LIKE_TO_INDEX_MAP.get(pauli_gate_like, None)
    if pauli_int is None:
        raise TypeError(
            f'Expected {key!r}: {pauli_gate_like!r} to have a '
            f'cirq.PAULI_GATE_LIKE value. '
            f"But the value isn't in "
            f"{set(PAULI_GATE_LIKE_TO_INDEX_MAP.keys())!r}"
        )
    return pauli_int
