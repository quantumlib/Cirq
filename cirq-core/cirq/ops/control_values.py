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
from typing import Collection, Optional, Sequence, Union, Tuple, List, Type, cast, TYPE_CHECKING

import copy
import itertools

if TYPE_CHECKING:
    import cirq


def flatten(sequence):
    def _flatten_aux(sequence):
        if isinstance(sequence, int):
            yield sequence
        else:
            for item in sequence:
                yield from _flatten_aux(item)

    return tuple(_flatten_aux(sequence))


def _from_int(val: int) -> Tuple[Tuple[int, ...], ...]:
    return ((val,),)


def _from_sequence_int(vals: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    return tuple((val,) for val in vals)


def _from_sequence_sequence(vals: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(product) for product in vals)


class ControlValues:
    def __init__(
        self, control_values: Sequence[Union[int, Collection[int], Type['ControlValues']]]
    ):
        if len(control_values) == 0:
            self._vals = cast(Tuple[Tuple[int, ...], ...], (()))
            self._num_variables = 0
            self._nxt = None
            return
        self._nxt = None

        if len(control_values) > 1:
            self._nxt = ControlValues(control_values[1:])

        if isinstance(control_values[0], ControlValues):
            aux = control_values[0].copy()
            aux.product(self._nxt)
            self._vals, self._num_variables, self._nxt = aux._vals, aux._num_variables, aux._nxt
            self._vals = cast(Tuple[Tuple[int, ...], ...], self._vals)
            return

        val = control_values[0]
        if isinstance(val, int):
            self._vals = _from_int(val)
        elif isinstance(val, (list, tuple)):
            if isinstance(val[0], int):
                self._vals = _from_sequence_int(val)
            else:
                self._vals = _from_sequence_sequence(val)
        else:
            raise TypeError(f'{val} is of Unsupported type {type(val)}')
        self._num_variables = len(self._vals[0])

    def product(self, other):
        """Sets self to be the cartesian product of all combinations in self x other.

        Args:
            other: A ControlValues object
        """
        if other is None:
            return
        other = other.copy()
        cur = self
        while cur._nxt is not None:
            cur = cur._nxt
        cur._nxt = other

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        nxt = self._nxt if self._nxt else lambda: [()]
        if self._num_variables:
            return itertools.product(self._vals, nxt())
        else:
            return itertools.product(*(), nxt())

    def copy(self):
        """Returns a deep copy of the object."""
        if self._num_variables == 0:
            new_copy = ControlValues([])
        else:
            new_copy = ControlValues(
                [
                    copy.deepcopy(self._vals),
                ]
            )
        new_copy._nxt = None
        if self._nxt:
            new_copy._nxt = self._nxt.copy()
        return new_copy

    def __len__(self):
        cur = self
        num_variables = 0
        while cur is not None:
            num_variables += cur._num_variables
            cur = cur._nxt
        return num_variables

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key != slice(None, -1, None):
                raise ValueError('Unsupported slicing')
            return self.copy().pop()
        key = int(key)
        num_variables = len(self)
        if not 0 <= key < num_variables:
            key = key % num_variables
        cur = self
        while cur._num_variables <= key:
            key -= cur._num_variables
            cur = cur._nxt
        return cur

    def __eq__(self, other):
        if not isinstance(other, ControlValues):
            try:
                other = ControlValues(other)
            except TypeError:
                return NotImplemented
        self_values = set(flatten(A) for A in self)
        other_values = set(flatten(B) for B in other)
        return self_values == other_values

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
            controls.append((cur._vals, companions[: cur._num_variables]))
            companions = companions[cur._num_variables :]
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
        if self._num_variables == 0:
            return
        if qid_shape is None and controls is None:
            raise ValueError('At least one of qid_shape or controls has to be not given.')
        if controls is not None:
            controls = tuple(controls)
        if (qid_shape is None or len(qid_shape) == 0) and controls is not None:
            qid_shape = tuple(q.dimension for q in controls[: self._num_variables])
        qid_shape = cast(Tuple[int], qid_shape)
        for product in self._vals:
            product = flatten(product)
            for i in range(self._num_variables):
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
                qid_shape=qid_shape[self._num_variables :],
                controls=controls[self._num_variables :] if controls else None,
                offset=offset + self._num_variables,
            )

    def are_same_value(self, value: int = 1):
        for product in self._vals:
            product = flatten(product)
            if not all(v == value for v in product):
                return False
        if self._nxt is not None:
            return self._nxt.are_same_value(value)
        return True

    def arrangements(self):
        """Returns a list of the control values.

        Returns:
            lists containing the control values whose product is the list of all
            possible combinations of the control values.
        """
        _arrangements = []
        cur = self
        while cur is not None:
            if cur._num_variables == 1:
                _arrangements.append(flatten(cur._vals))
            else:
                _arrangements.append(tuple(flatten(product) for product in cur._vals))
            cur = cur._nxt
        return _arrangements

    def pop(self):
        """Removes the last control values combination."""
        if self._nxt is None:
            return None
        self._nxt = self._nxt.pop()
        return self


def to_control_values(
    values: Union[ControlValues, Sequence[Union[int, Collection[int]]]]
) -> ControlValues:
    if not isinstance(values, ControlValues):
        # Convert to sorted tuples
        return ControlValues(
            tuple((val,) if isinstance(val, int) else tuple(sorted(val)) for val in values)
        )
    else:
        return values


class FreeVars(ControlValues):
    pass


class ConstrainedVars(ControlValues):
    def __init__(self, control_values):
        sum_of_product = (tuple(zip(*control_values)),)
        super().__init__(sum_of_product)
