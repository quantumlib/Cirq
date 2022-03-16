# Copyright 2021 The Cirq Developers
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
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Tuple, List, Type, cast, TYPE_CHECKING, Any, Dict

import itertools

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True, eq=False)  # type: ignore
class ControlValues(abc.ABC):
    """ControlValues is an abstract data class that represents different control values."""

    _combinations: Any
    _nxt: Optional['ControlValues'] = None

    def product(self, other) -> 'ControlValues':
        """Sets self to be the cartesian product of all combinations in self x other.

        Args:
            other: A ControlValues object
        """
        return self.link(other)

    @staticmethod
    def flatten(sequence: Sequence[Union[int, Sequence]]) -> Tuple[int, ...]:
        """Takes a sequence and recursively flattens it.

        Args:
            sequence: An iterable to flatten

        Returns:
            A flattened tuple.
        """

        def _flatten_aux(sequence):
            if isinstance(sequence, int):
                yield sequence
            else:
                for item in sequence:
                    yield from _flatten_aux(item)

        return tuple(_flatten_aux(sequence))

    def __call__(self):
        return self.__iter__()

    @abc.abstractmethod
    def expand(self):
        """Returns the control values tracked by the object."""

    @abc.abstractmethod
    def num_variables(self) -> int:
        """Returns number of variables tracked by the object."""

    def __iter__(self):
        nxt = self._nxt if self._nxt else lambda: [()]
        return itertools.product(self.expand(), nxt())

    def __len__(self) -> int:
        cur = cast(Optional['ControlValues'], self)
        num_variables = 0
        while cur is not None:
            num_variables += cur.num_variables()
            cur = cur._nxt
        return num_variables

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key != slice(None, -1, None):
                raise ValueError('Unsupported slicing')
            return self.pop()
        key = int(key)
        num_variables = len(self)
        if not 0 <= key < num_variables:
            key = key % num_variables
        cur = cast(Optional['ControlValues'], self)
        while cur.num_variables() <= key:
            key -= cur.num_variables()
            cur = cur._nxt
        return cur

    @abc.abstractmethod
    def __eq__(self, other):
        """Returns True iff self and other represent the same configurations.

        Args:
            other: A ControlValues object.

        Returns:
            boolean whether the two objects are equivalent or not.
        """

    def pop(self) -> Optional['ControlValues']:
        """Removes the last control values combination."""
        if self._nxt is None:
            return None
        return type(self)(self._combinations, self._nxt.pop())

    def link(self, other) -> 'ControlValues':
        """Links two ControlValues sequences.

        Args:
            other: An instance of a descendent type of ControlValues

        Returns:
            A new object that represents the product of the combinations in the two objects.
        """
        if self._nxt is None:
            return type(self)(self._combinations, other)
        return type(self)(self._combinations, self._nxt.link(other))

    @abc.abstractmethod
    def identifier(self, companions: Sequence[Union[int, 'cirq.Qid']]):
        """Attaches the companions to their control values (used in comparisons).

        Args:
            companions: a sequence of Qid or ints.

        Returns:
            A sequence of zipped companions to their control values.
        """

    @abc.abstractmethod
    def check_dimensionality(
        self,
        qid_shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        controls: Optional[Union[Tuple['cirq.Qid', ...], List['cirq.Qid']]] = None,
        offset=0,
    ) -> None:
        """Checks the dimentionality of control values with respect to qid_shape or controls.
            *At least one of qid_shape and controls must be provided.
            *if both are provided then controls is ignored*

        Args:
            qid_shape: Sequence of shapes of the control qubits.
            controls: Sequence of Qids.
            offset: starting index.
        Raises:
            ValueError:
                - if none of qid_shape or controls are provided or both are empty.
                - if one of the control values violates the shape of its qubit.
        """

    @abc.abstractmethod
    def are_same_value(self, target_value) -> bool:
        """Checks whether all control values are the same and equal target_value.

        Args:
            target_value: a value to check whether all control values are eqaul to or not.

        Returns:
            True/False whether all same control values equal target_value.
        """

    @abc.abstractmethod
    def arrangements(self):
        """Returns a representation of the control values,
        The representation can be used in comparisons and to rebuild the object."""

    @staticmethod
    @abc.abstractmethod
    def factory(val) -> 'ControlValues':
        """Builds a ControlValue from values."""

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            '_combinations': self._combinations,
            '_nxt': self._nxt._json_dict_() if self._nxt else None,
        }


@ControlValues.register
class SimpleControlValues(ControlValues):
    """SimpleControlValues represents control values in a form of tables of integers."""

    _combinations: Tuple[Tuple[int, ...], ...] = ((),)

    def expand(self):
        """Returns the combinations tracked by the object."""
        return self._combinations

    def num_variables(self) -> int:
        """Returns number of variables tracked by the object."""
        return len(self._combinations[0])

    def identifier(self, companions: Sequence[Union[int, 'cirq.Qid']]):
        """Returns an identifier that pairs the control_values with their counterparts
            in the companion sequence (e.g. sequence of the control qubits ids)

        Args:
            companions: sequence of the same length as the number of control qubits.

        Returns:
            Tuple of pairs of the control values paired with its corresponding companion.
        """
        companions = tuple(companions)
        controls = []
        cur = cast(Optional[ControlValues], self)
        while cur is not None:
            controls.append((cur._combinations, companions[: cur.num_variables()]))
            companions = companions[cur.num_variables() :]
            cur = cur._nxt
        return tuple(controls)

    def check_dimensionality(
        self,
        qid_shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        controls: Optional[Union[Tuple['cirq.Qid', ...], List['cirq.Qid']]] = None,
        offset=0,
    ):
        """Checks the dimentionality of control values with respect to qid_shape or controls.
            *At least one of qid_shape and controls must be provided.
            *if both are provided then controls is ignored*

        Args:
            qid_shape: Sequence of shapes of the control qubits.
            controls: Sequence of Qids.
            offset: starting index.
        Raises:
            ValueError:
                - if none of qid_shape or controls are provided or both are empty.
                - if one of the control values violates the shape of its qubit.
        """
        if self.num_variables() == 0:
            return
        if qid_shape is None and controls is None:
            raise ValueError('At least one of qid_shape or controls has to be not given.')
        if controls is not None:
            controls = tuple(controls)
        if (qid_shape is None or len(qid_shape) == 0) and controls is not None:
            qid_shape = tuple(q.dimension for q in controls[: self.num_variables()])
        qid_shape = cast(Tuple[int], qid_shape)
        for product in self._combinations:
            product = ControlValues.flatten(product)
            for i in range(self.num_variables()):
                if not 0 <= product[i] < qid_shape[i]:
                    message = (
                        'Control values <{!r}> outside of range ' 'for control qubit number <{!r}>.'
                    ).format(product[i], i + offset)
                    if controls is not None:
                        message = (
                            f'Control values <{product[i]!r}> outside of range'
                            f' for qubit <{controls[i]!r}>.'
                        )
                    raise ValueError(message)

        if self._nxt is not None:
            self._nxt.check_dimensionality(
                qid_shape=qid_shape[self.num_variables() :],
                controls=controls[self.num_variables() :] if controls else None,
                offset=offset + self.num_variables(),
            )

    def are_same_value(self, target_value: int = 1) -> bool:
        """Checks whether all control values are the same and equal target_value.

        Args:
            target_value: a value to check whether all control values are eqaul to or not.

        Returns:
            True/False whether all same control values equal target_value.
        """
        for product in self._combinations:
            product = ControlValues.flatten(product)
            if not all(v == target_value for v in product):
                return False
        if self._nxt is not None:
            return self._nxt.are_same_value(target_value)
        return True

    def arrangements(self):
        """Returns a tuple of the control values.

        Returns:
            lists containing the control values whose product is the list of all
            possible combinations of the control values.
        """
        _arrangements = []
        cur = self
        while cur is not None:
            if cur.num_variables() <= 1:
                _arrangements.append(ControlValues.flatten(cur._combinations))
            else:
                _arrangements.append(
                    tuple(ControlValues.flatten(product) for product in cur._combinations)
                )
            cur = cur._nxt
        return tuple(_arrangements)

    @classmethod
    def from_int(cls, val: int) -> 'SimpleControlValues':
        """Builds a SimpleControlValues from an int.

        Args:
            val: value to build a SimpleControlValues around.

        Returns:
            SimpleControlValues
        """
        return cls(((val,),))

    @classmethod
    def from_sequence_int(cls, vals: Sequence[int]) -> 'SimpleControlValues':
        """Builds a SimpleControlValues from a sequence of integers.

        Args:
            vals: a sequence of ints.

        Returns:
            SimpleControlValues
        """
        return cls(tuple((val,) for val in sorted(vals)))

    @classmethod
    def from_sequence_sequence(cls, vals: Sequence[Sequence[int]]) -> 'SimpleControlValues':
        """Builds a SimpleControlValues from a sequence of sequences of integers.

        Args:
            vals: a sequence of sequences of ints.

        Returns:
            SimpleControlValues
        """
        return cls(tuple(tuple(product) for product in vals))

    @staticmethod
    def factory(val) -> ControlValues:
        """Builds a SimpleControlValues from any if athe allowed types.

        Args:
            val: an int, Sequence[int], or a Sequence[Sequence[int]]

        Returns:
            SimpleControlValues

        Raises:
            TypeError: if the type of val is not supported.
        """
        if isinstance(val, int):
            return SimpleControlValues.from_int(val)
        if isinstance(val, (list, tuple)):
            if isinstance(val[0], int):
                return SimpleControlValues.from_sequence_int(val)
            return SimpleControlValues.from_sequence_sequence(val)
        raise TypeError(f'{val} is of Unsupported type {type(val)}')

    def __eq__(self, other) -> bool:
        """Returns True iff self and other represent the same configurations.

        Args:
            other: A ControlValues object.

        Returns:
            boolean whether the two objects are equivalent or not.
        """
        if not isinstance(other, ControlValues):
            if isinstance(other, (list, tuple)):
                if isinstance(other[0], int):
                    return self == SimpleControlValues.from_sequence_int(other)
                return self.arrangements() == tuple(sorted(other))
            return NotImplemented
        self_values = set(ControlValues.flatten(A) for A in self)
        other_values = set(ControlValues.flatten(B) for B in other)
        return self_values == other_values


@SimpleControlValues.register
class ConstrainedValues(SimpleControlValues):
    """ConstrainedValues represents configurations of control values.

    Example:
        ConstrainedValues.factory([[0, 1], [1, 0]]) represents the xor of two qubits.
    """

    @classmethod
    def from_values(cls, control_values: Sequence[Sequence[int]]) -> SimpleControlValues:
        """Builds a ConstrainedValues from a sequence of sequences of integers.

        Args:
            control_values: a sequence of sequences of ints.

        Returns:
            SimpleControlValues
        """
        return cls(tuple(zip(*control_values)))

    @staticmethod
    def factory(val) -> ControlValues:
        return ConstrainedValues.from_values(val)


class ControlValuesBuilder:
    """Builds a ControlValues structure."""

    def __init__(self):
        self.chain = []

    def add_combinations(
        self, vals: Any, control_type: Type['ControlValues'] = SimpleControlValues
    ) -> 'ControlValuesBuilder':
        """Inserts vals at the end of the structure after building it as a control_type.

        Args:
            vals: Either a control values object or values to use to create a ControlValues object.
            control_type: A child class of ControlValues to use to build the object.
        """
        if not isinstance(vals, ControlValues):
            vals = control_type.factory(vals)
        self.chain.append(vals)
        return self

    def append(
        self,
        vals: Union[ControlValues, Sequence[Any]],
        control_type: Type['ControlValues'] = SimpleControlValues,
    ) -> 'ControlValuesBuilder':
        """Inserts a sequences to the end of the structure.

        Args:
            vals: Either a ControlValues object or a sequence of values to create ControlValues.
            control_type: A child class of ControlValues to use to build the object.
        """
        if isinstance(vals, ControlValues):
            self.add_combinations(vals)
        else:
            for val in vals:
                self.add_combinations(val, control_type)
        return self

    def build(self) -> ControlValues:
        """Builds a ControlValues representing all control values present in the object."""
        if len(self.chain) == 0:
            return SimpleControlValues(((),), None)
        lst = None
        for val in self.chain[::-1]:
            lst = val.link(lst)
        lst = cast(ControlValues, lst)
        return lst
