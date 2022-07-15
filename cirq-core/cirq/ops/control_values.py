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
from typing import Union, Tuple, List, TYPE_CHECKING, Any, Dict, Generator, cast, Iterator, Optional
from dataclasses import dataclass

import itertools

if TYPE_CHECKING:
    import cirq


class AbstractControlValues(abc.ABC):
    """Abstract base class defining the API for control values.

    `AbstractControlValues` is an abstract class that defines the API for control values
    and implements functions common to all implementations (e.g.  comparison).

    `cirq.ControlledGate` and `cirq.ControlledOperation` are useful to augment
    existing gates and operations to have one or more control qubits. For every
    control qubit, the set of integer values for which the control should be enabled
    is represented by one of the implementations of `cirq.AbstractControlValues`.

    Implementations of `cirq.AbstractControlValues` can use different internal
    representations to store control values, but they must satisfy the public API
    defined here and be immutable.
    """

    @abc.abstractmethod
    def __and__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Sets self to be the cartesian product of all combinations in self x other.

        Args:
          other: An object that implements AbstractControlValues.

        Returns:
          An object that represents the cartesian product of the two inputs.
        """

    @abc.abstractmethod
    def _expand(self) -> Iterator[Tuple[int, ...]]:
        """Expands the (possibly compressed) internal representation into a sum of products representation."""  # pylint: disable=line-too-long

    @abc.abstractmethod
    def diagram_repr(self, label: Optional[str] = None) -> str:
        """Returns a string representation to be used in circuit diagrams."""

    @abc.abstractmethod
    def _number_variables(self) -> int:
        """Returns the number of variables controlled by the object."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def _identifier(self) -> Any:
        """Returns the internal representation of the object."""

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def validate(self, qid_shapes: Union[Tuple[int, ...], List[int]]) -> None:
        """Validates control values

        Validate that control values are in the half closed interval
        [0, qid_shapes) for each qubit.
        """

    @abc.abstractmethod
    def _are_ones(self) -> bool:
        """Checks whether all control values are equal to 1."""

    @abc.abstractmethod
    def _json_dict_(self) -> Dict[str, Any]:
        pass

    def __iter__(self) -> Generator[Tuple[int, ...], None, None]:
        for assignment in self._expand():
            yield assignment

    def __eq__(self, other) -> bool:
        """Returns True iff self and other represent the same configurations.

        Args:
            other: A AbstractControlValues object.

        Returns:
            boolean whether the two objects are equivalent or not.
        """
        if not isinstance(other, AbstractControlValues):
            other = ProductOfSums(other)
        return sorted(v for v in self) == sorted(v for v in other)


@dataclass(frozen=True, eq=False)
class ProductOfSums(AbstractControlValues):
    """ProductOfSums represents control values in a form of a cartesian product of tuples."""

    _internal_representation: Tuple[Tuple[int, ...], ...]

    def _identifier(self) -> Tuple[Tuple[int, ...], ...]:
        return self._internal_representation

    def _expand(self) -> Iterator[Tuple[int, ...]]:
        """Returns the combinations tracked by the object."""
        self = cast('ProductOfSums', self)
        return itertools.product(*self._internal_representation)

    def __repr__(self) -> str:
        return f'cirq.ProductOfSums({str(self._identifier())})'

    def _number_variables(self) -> int:
        return len(self._internal_representation)

    def __len__(self) -> int:
        return self._number_variables()

    def __hash__(self) -> int:
        return hash(self._internal_representation)

    def validate(self, qid_shapes: Union[Tuple[int, ...], List[int]]) -> None:
        for i, (vals, shape) in enumerate(zip(self._internal_representation, qid_shapes)):
            if not all(0 <= v < shape for v in vals):
                message = (
                    f'Control values <{vals!r}> outside of range for control qubit '
                    f'number <{i}>.'
                )
                raise ValueError(message)

    def _are_ones(self) -> bool:
        return frozenset(self._internal_representation) == {(1,)}

    def diagram_repr(self, label: Optional[str] = None) -> str:
        if label:
            return label
        if self._are_ones():
            return 'C' * self._number_variables()

        def get_prefix(control_vals):
            control_vals_str = ''.join(map(str, sorted(control_vals)))
            return f'C{control_vals_str}'

        return ''.join(map(get_prefix, self._internal_representation))

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union['AbstractControlValues', Tuple[int, ...]]:
        if isinstance(key, slice):
            return ProductOfSums(self._internal_representation[key])
        return self._internal_representation[key]

    def _json_dict_(self) -> Dict[str, Any]:
        return {'_internal_representation': self._internal_representation}

    def __and__(self, other: AbstractControlValues) -> AbstractControlValues:
        if isinstance(other, SumOfProducts):
            return SumOfProducts(tuple(p for p in self)) & other
        if not isinstance(other, ProductOfSums):
            raise TypeError(
                f'And operation not supported between types ProductOfSums and {type(other)}'
            )
        return type(self)(self._internal_representation + other._internal_representation)


@dataclass(frozen=True, eq=False)
class SumOfProducts(AbstractControlValues):
    """Represents control values as a union of n-bit tuples.

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
        >>> nan_cop = cirq.X(q2).controlled_by(q0, q1, control_values=nand_control_values)
    """

    _internal_representation: Tuple[Tuple[int, ...], ...]

    def __post_init__(self):
        if not len(self._internal_representation):
            raise ValueError('SumOfProducts can\'t be empty.')
        num_qubits = len(self._internal_representation[0])
        for p in self._internal_representation:
            if len(p) != num_qubits:
                raise ValueError(
                    f'size mismatch between different products of {self._internal_representation}'
                )
        if len(self._internal_representation) != len(
            set(map(tuple, self._internal_representation))
        ):
            raise ValueError('SumOfProducts can\'t have duplicate products.')

    def _identifier(self) -> Tuple[Tuple[int, ...], ...]:
        return self._internal_representation

    def _expand(self) -> Iterator[Tuple[int, ...]]:
        """Returns the combinations tracked by the object."""
        self = cast('SumOfProducts', self)
        return iter(self._internal_representation)

    def __repr__(self) -> str:
        return f'cirq.SumOfProducts({str(self._identifier())})'

    def _number_variables(self) -> int:
        return len(self._internal_representation[0])

    def __len__(self) -> int:
        return self._number_variables()

    def __hash__(self) -> int:
        return hash(self._internal_representation)

    def validate(self, qid_shapes: Union[Tuple[int, ...], List[int]]) -> None:
        for vals in self._internal_representation:
            if len(qid_shapes) != len(vals):
                raise ValueError(
                    f'number of values in product {vals} doesn\'t  equal number'
                    f' of qubits(={len(qid_shapes)})'
                )

            for i, v in enumerate(vals):
                if not (0 <= v and v < qid_shapes[i]):
                    raise ValueError(
                        f'Control values <{v}> in combination {vals} is outside'
                        f' of range for control qubit number <{i}>.'
                    )

    def _are_ones(self) -> bool:
        return frozenset(self._internal_representation) == {(1,) * self._number_variables()}

    def diagram_repr(self, label: Optional[str] = None) -> str:
        if label:
            return label
        return ','.join(map(lambda p: ''.join(map(str, p)), self._internal_representation))

    def _json_dict_(self) -> Dict[str, Any]:
        return {'_internal_representation': self._internal_representation}

    def __and__(self, other: AbstractControlValues) -> 'SumOfProducts':
        if isinstance(other, ProductOfSums):
            other = SumOfProducts(tuple(p for p in other))
        if not isinstance(other, SumOfProducts):
            raise TypeError(
                f'And operation not supported between types SumOfProducts and {type(other)}'
            )
        combined = map(
            lambda p: tuple(itertools.chain(*p)),
            itertools.product(self._internal_representation, other._internal_representation),
        )
        return SumOfProducts(tuple(combined))
