# Copyright 2022 The Cirq Developers
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
import abc
from functools import cached_property
from typing import Collection, Tuple, TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Union
import itertools

from cirq import protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality
class AbstractControlValues(abc.ABC):
    """Abstract base class defining the API for control values.

    Control values define predicates on the state of one or more qubits. Predicates can be composed
    with logical OR to form a "sum", or with logical AND to form a "product". We provide two
    implementations: `SumOfProducts` which consists of one or more AND (product) clauses each of
    which applies to all N qubits, and `ProductOfSums` which consists of N OR (sum) clauses,
    each of which applies to one qubit.

    `cirq.ControlledGate` and `cirq.ControlledOperation` are useful to augment
    existing gates and operations to have one or more control qubits. For every
    control qubit, the set of integer values for which the control should be enabled
    is represented by one of the implementations of `cirq.AbstractControlValues`.

    Implementations of `cirq.AbstractControlValues` can use different internal
    representations to store control values, but they must satisfy the public API
    defined here and be immutable.
    """

    @abc.abstractmethod
    def validate(self, qid_shapes: Sequence[int]) -> None:
        """Validates that all control values for ith qubit are in range [0, qid_shaped[i])"""

    @abc.abstractmethod
    def expand(self) -> 'SumOfProducts':
        """Returns an expanded `cirq.SumOfProduct` representation of this control values."""

    @property
    @abc.abstractmethod
    def is_trivial(self) -> bool:
        """Returns True iff each controlled variable is activated only for value 1.

        This configuration is equivalent to `cirq.SumOfProducts(((1,) * num_controls))`
        and `cirq.ProductOfSums(((1,),) * num_controls)`
        """

    @abc.abstractmethod
    def _num_qubits_(self) -> int:
        """Returns the number of qubits for which control values are stored by this object."""

    @abc.abstractmethod
    def _json_dict_(self) -> Dict[str, Any]:
        """Returns a dictionary used for serializing this object."""

    @abc.abstractmethod
    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Returns information used to draw this object in circuit diagrams."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Iterator on internal representation of control values used by the derived classes.

        Note: Be careful that the terms iterated upon by this iterator will have different
        meaning based on the implementation. For example:
        >>> print(*cirq.ProductOfSums([(0, 1), (0,)]))
        (0, 1) (0,)
        >>> print(*cirq.SumOfProducts([(0, 0), (1, 0)]))
        (0, 0) (1, 0)
        """

    def _value_equality_values_(self) -> Any:
        return tuple(v for v in self.expand())

    def __and__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Returns a cartesian product of all control values predicates in `self` x `other`.

        The `and` of two control values `cv1` and `cv2` represents a control value object
        acting on the union of qubits represented by `cv1` and `cv2`. For example:

        >>> cv1 = cirq.ProductOfSums([(0, 1), 2])
        >>> cv2 = cirq.SumOfProducts([[0, 0], [1, 1]])
        >>> assert cirq.num_qubits(cv1 & cv2) == cirq.num_qubits(cv1) + cirq.num_qubits(cv2)

        Args:
          other: An instance of `AbstractControlValues`.

        Returns:
          An instance of `AbstractControlValues` that represents the cartesian product of
          control values represented by `self` and `other`.
        """
        return SumOfProducts(
            tuple(x + y for (x, y) in itertools.product(self.expand(), other.expand()))
        )

    def __or__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Returns a union of all control values predicates in `self` + `other`.

        Both `self` and `other` must represent control values for the same set of qubits and
        hence their `or` would also be a control value object acting on the same set of qubits.
        For example:

        >>> cv1 = cirq.ProductOfSums([(0, 1), 2])
        >>> cv2 = cirq.SumOfProducts([[0, 0], [1, 1]])
        >>> assert cirq.num_qubits(cv1 | cv2) == cirq.num_qubits(cv1) == cirq.num_qubits(cv2)

        Args:
          other: An instance of `AbstractControlValues`.

        Returns:
          An instance of `AbstractControlValues` that represents the union of control values
          represented by `self` and `other`.

        Raises:
            ValueError: If `cirq.num_qubits(self) != cirq.num_qubits(other)`.
        """
        if protocols.num_qubits(self) != protocols.num_qubits(other):
            raise ValueError(
                f"Control values {self} and {other} must act on equal number of qubits"
            )
        return SumOfProducts((*self.expand(), *other.expand()))


class ProductOfSums(AbstractControlValues):
    """Represents control values as N OR (sum) clauses, each of which applies to one qubit."""

    def __init__(self, data: Sequence[Union[int, Collection[int]]]):
        self._qubit_sums: Tuple[Tuple[int, ...], ...] = tuple(
            (cv,) if isinstance(cv, int) else tuple(sorted(set(cv))) for cv in data
        )

    @cached_property
    def is_trivial(self) -> bool:
        return self._qubit_sums == ((1,),) * self._num_qubits_()

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        return iter(self._qubit_sums)

    def expand(self) -> 'SumOfProducts':
        return SumOfProducts(tuple(itertools.product(*self._qubit_sums)))

    def __repr__(self) -> str:
        return f'cirq.ProductOfSums({str(self._qubit_sums)})'

    def _num_qubits_(self) -> int:
        return len(self._qubit_sums)

    def __getitem__(self, key: Union[int, slice]) -> Union['ProductOfSums', Tuple[int, ...]]:
        if isinstance(key, slice):
            return ProductOfSums(self._qubit_sums[key])
        return self._qubit_sums[key]

    def validate(self, qid_shapes: Sequence[int]) -> None:
        for i, (vals, shape) in enumerate(zip(self._qubit_sums, qid_shapes)):
            if not all(0 <= v < shape for v in vals):
                message = (
                    f'Control values <{vals!r}> outside of range for control qubit '
                    f'number <{i}>.'
                )
                raise ValueError(message)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Returns a string representation to be used in circuit diagrams."""

        def get_symbol(vals):
            return '@' if tuple(vals) == (1,) else f"({','.join(map(str, vals))})"

        return protocols.CircuitDiagramInfo(wire_symbols=[get_symbol(t) for t in self._qubit_sums])

    def __str__(self) -> str:
        if self.is_trivial:
            return 'C' * self._num_qubits_()

        def get_prefix(control_vals):
            control_vals_str = ''.join(map(str, sorted(control_vals)))
            return f'C{control_vals_str}'

        return ''.join(get_prefix(t) for t in self._qubit_sums)

    def _json_dict_(self) -> Dict[str, Any]:
        return {"data": self._qubit_sums}

    def __and__(self, other: AbstractControlValues) -> AbstractControlValues:
        if isinstance(other, ProductOfSums):
            return ProductOfSums(self._qubit_sums + other._qubit_sums)
        return super().__and__(other)

    def __or__(self, other: AbstractControlValues) -> AbstractControlValues:
        if protocols.num_qubits(self) != protocols.num_qubits(other):
            raise ValueError(
                f"Control values {self} and {other} must act on equal number of qubits"
            )
        if isinstance(other, ProductOfSums):
            return ProductOfSums(tuple(x + y for x, y in zip(self._qubit_sums, other._qubit_sums)))
        return super().__or__(other)


class SumOfProducts(AbstractControlValues):
    """Represents control values as AND (product) clauses, each of which applies to all N qubits.

    `SumOfProducts` representation describes the control values as a union
    of n-bit tuples, where each n-bit tuple represents an allowed assignment
    of bits for which the control should be activated. This expanded
    representation allows us to create control values combinations which
    cannot be factored as a `ProductOfSums` representation.

    For example:

    1) `(|00><00| + |11><11|) X + (|01><01| + |10><10|) I` represents an
        operator which flips the third qubit if the first two qubits
        are `00` or `11`, and does nothing otherwise.
        This can be constructed as
        >>> xor_control_values = cirq.SumOfProducts(((0, 0), (1, 1)))
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> xor_cop = cirq.X(q2).controlled_by(q0, q1, control_values=xor_control_values)

    2) `(|00><00| + |01><01| + |10><10|) X + (|11><11|) I` represents an
        operators which flips the third qubit if the `nand` of first two
        qubits is `1` (i.e. first two qubits are either `00`, `01` or `10`),
        and does nothing otherwise. This can be constructed as:

        >>> nand_control_values = cirq.SumOfProducts(((0, 0), (0, 1), (1, 0)))
        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> nand_cop = cirq.X(q2).controlled_by(q0, q1, control_values=nand_control_values)
    """

    def __init__(self, data: Collection[Sequence[int]], *, name: Optional[str] = None):
        self._conjunctions: Tuple[Tuple[int, ...], ...] = tuple(
            sorted(set(tuple(cv) for cv in data))
        )
        self._name = name
        if not len(self._conjunctions):
            raise ValueError("SumOfProducts can't be empty.")
        num_qubits = len(self._conjunctions[0])
        if not all(len(p) == num_qubits for p in self._conjunctions):
            raise ValueError(f'Each term of {self._conjunctions} should be of length {num_qubits}.')

    @cached_property
    def is_trivial(self) -> bool:
        return self._conjunctions == ((1,) * self._num_qubits_(),)

    def expand(self) -> 'SumOfProducts':
        return self

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Returns the combinations tracked by the object."""
        return iter(self._conjunctions)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Returns a string representation to be used in circuit diagrams."""
        if self._name is not None:
            wire_symbols = ['@'] * self._num_qubits_()
            wire_symbols[-1] = f'@({self._name})'
            return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)

        if len(self._conjunctions) == 1:
            # Use a simpler diagram if there's only 1 term.
            return protocols.CircuitDiagramInfo(
                wire_symbols=["@" if x == 1 else f"({x})" for x in self._conjunctions[0]]
            )

        wire_symbols = [''] * self._num_qubits_()
        for term in self._conjunctions:
            for q_i, q_val in enumerate(term):
                wire_symbols[q_i] = wire_symbols[q_i] + str(q_val)
        wire_symbols = [f'@({s})' for s in wire_symbols]
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __repr__(self) -> str:
        name = '' if self._name is None else f', name="{self._name}"'
        return f'cirq.SumOfProducts({self._conjunctions!s} {name})'

    def __str__(self) -> str:
        suffix = (
            self._name
            if self._name is not None
            else '_'.join(''.join(str(v) for v in t) for t in self._conjunctions)
        )
        return f'C_{suffix}'

    def _num_qubits_(self) -> int:
        return len(self._conjunctions[0])

    def validate(self, qid_shapes: Sequence[int]) -> None:
        if len(qid_shapes) != self._num_qubits_():
            raise ValueError(
                f'Length of qid_shapes: {qid_shapes} should be equal to self._num_qubits_():'
                f' {self._num_qubits_()}'
            )

        for product in self._conjunctions:
            for q_i, q_val in enumerate(product):
                if not (0 <= q_val < qid_shapes[q_i]):
                    raise ValueError(
                        f'Control value <{q_val}> in combination {product} is outside'
                        f' of range [0, {qid_shapes[q_i]}) for control qubit number <{q_i}>.'
                    )

    def _json_dict_(self) -> Dict[str, Any]:
        return {'data': self._conjunctions, 'name': self._name}
