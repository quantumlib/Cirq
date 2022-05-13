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
from typing import Union, Tuple, List, TYPE_CHECKING, Any, Dict
from dataclasses import dataclass

import itertools

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True, eq=False)  # type: ignore
class AbstractControlValues(abc.ABC):
    """AbstractControlValues is an abstract immutable data class.

    AbstractControlValues defines an API for control values and implements
    functions common to all implementations (e.g.  comparison).
    """

    _internal_representation: Any

    def __and__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Sets self to be the cartesian product of all combinations in self x other.

        Args:
          other: An object that implements AbstractControlValues.

        Returns:
          An object that represents the cartesian product of the two inputs.
        """
        return type(self)(self._internal_representation + other._internal_representation)

    def _iterator(self):
        return self._expand()

    @abc.abstractmethod
    def _expand(self):
        """Returns the control values tracked by the object."""

    @abc.abstractmethod
    def diagram_repr(self) -> str:
        """Returns a string representation to be used in circuit diagrams."""

    @abc.abstractmethod
    def _number_variables(self):
        """Returns the control values tracked by the object."""

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def identifier(self) -> Tuple[Any]:
        """Returns an identifier from which the object can be rebuilt."""

    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def _validate(self, qid_shapes: Union[Tuple[int, ...], List[int]]) -> None:
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

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    def __iter__(self):
        for assignment in self._iterator():
            yield assignment

    def __eq__(self, other):
        """Returns True iff self and other represent the same configurations.

        Args:
            other: A AbstractControlValues object.

        Returns:
            boolean whether the two objects are equivalent or not.
        """
        if not isinstance(other, AbstractControlValues):
            other = ProductOfSums(other)
        return sorted(v for v in self) == sorted(v for v in other)


@AbstractControlValues.register
class ProductOfSums(AbstractControlValues):
    """ProductOfSums represents control values in a form of a cartesian product of tuples."""

    _internal_representation: Tuple[Tuple[int, ...]]

    def identifier(self):
        return self._internal_representation

    def _expand(self):
        """Returns the combinations tracked by the object."""
        return itertools.product(*self._internal_representation)

    def __repr__(self):
        return f'cirq.ProductOfSums({str(self.identifier())})'

    def _number_variables(self) -> int:
        return len(self._internal_representation)

    def __len__(self):
        return self._number_variables()

    def __hash__(self):
        return hash(self._internal_representation)

    def _validate(self, qid_shapes: Union[Tuple[int, ...], List[int]]) -> None:
        for i, (vals, shape) in enumerate(zip(self._internal_representation, qid_shapes)):
            if not all(0 <= v < shape for v in vals):
                message = (
                    f'Control values <{vals!r}> outside of range for control qubit '
                    f'number <{i}>.'
                )
                raise ValueError(message)

    def _are_ones(self) -> bool:
        return frozenset(self._internal_representation) == {(1,)}

    def diagram_repr(self) -> str:
        if self._are_ones():
            return 'C' * self._number_variables()

        def get_prefix(control_vals):
            control_vals_str = ''.join(map(str, sorted(control_vals)))
            return f'C{control_vals_str}'

        return ''.join(map(get_prefix, self._internal_representation))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ProductOfSums(self._internal_representation[key])
        return self._internal_representation[key]

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            '_internal_representation': self._internal_representation,
            'cirq_type': 'ProductOfSums',
        }
