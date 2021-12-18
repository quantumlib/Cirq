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
            self.vals = cast(Tuple[Tuple[int, ...], ...], (()))
            self.num_variables = 0
            self.nxt = None
            self.itr = None
            return
        self.itr = None
        self.nxt = None

        if len(control_values) > 1:
            self.nxt = ControlValues(control_values[1:])

        if isinstance(control_values[0], ControlValues):
            aux = control_values[0].copy()
            aux.product(self.nxt)
            self.vals, self.num_variables, self.nxt = aux.vals, aux.num_variables, aux.nxt
            self.vals = cast(Tuple[Tuple[int, ...], ...], self.vals)
            return

        val = control_values[0]
        if isinstance(val, int):
            self.vals = _from_int(val)
        elif isinstance(val, (list, tuple)):
            if isinstance(val[0], int):
                self.vals = _from_sequence_int(val)
            else:
                self.vals = _from_sequence_sequence(val)
        else:
            raise TypeError(f'{val} is of Unsupported type {type(val)}')
        self.num_variables = len(self.vals[0])

    def product(self, other):
        # Cartesian product of all combinations in self x other
        if other is None:
            return
        other = other.copy()
        cur = self
        while cur.nxt is not None:
            cur = cur.nxt
        cur.nxt = other

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        nxt = self.nxt if self.nxt else lambda: [()]
        if self.num_variables:
            self.itr = itertools.product(self.vals, nxt())
        else:
            self.itr = itertools.product(*(), nxt())
        return self.itr

    def copy(self):
        if self.num_variables == 0:
            new_copy = ControlValues([])
        else:
            new_copy = ControlValues(
                [
                    copy.deepcopy(self.vals),
                ]
            )
        new_copy.nxt = None
        if self.nxt:
            new_copy.nxt = self.nxt.copy()
        return new_copy

    def __len__(self):
        cur = self
        num_variables = 0
        while cur is not None:
            num_variables += cur.num_variables
            cur = cur.nxt
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
        while cur.num_variables <= key:
            key -= cur.num_variables
            cur = cur.nxt
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
        companions = tuple(companions)
        controls = []
        cur = cast(Optional[ControlValues], self)
        while cur is not None:
            controls.append((cur.vals, companions[: cur.num_variables]))
            companions = companions[cur.num_variables :]
            cur = cur.nxt
        return tuple(controls)

    def check_dimentionality(
        self,
        qid_shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        controls: Optional[Union[Tuple['cirq.Qid', ...], List['cirq.Qid']]] = None,
        offset=0,
    ):
        if self.num_variables == 0:
            return
        if qid_shape is None and controls is None:
            raise ValueError('At least one of qid_shape or controls has to be not given.')
        if controls is not None:
            controls = tuple(controls)
        if (qid_shape is None or len(qid_shape) == 0) and controls is not None:
            qid_shape = tuple(q.dimension for q in controls[: self.num_variables])
        qid_shape = cast(Tuple[int], qid_shape)
        for product in self.vals:
            product = flatten(product)
            for i in range(self.num_variables):
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

        if self.nxt is not None:
            self.nxt.check_dimentionality(
                qid_shape=qid_shape[self.num_variables :],
                controls=controls[self.num_variables :] if controls else None,
                offset=offset + self.num_variables,
            )

    def are_same_value(self, value: int = 1):
        for product in self.vals:
            product = flatten(product)
            if not all(v == value for v in product):
                return False
        if self.nxt is not None:
            return self.nxt.are_same_value(value)
        return True

    def arrangements(self):
        _arrangements = []
        cur = self
        while cur is not None:
            if cur.num_variables == 1:
                _arrangements.append(flatten(cur.vals))
            else:
                _arrangements.append(tuple(flatten(product) for product in cur.vals))
            cur = cur.nxt
        return _arrangements

    def pop(self):
        if self.nxt is None:
            return None
        self.nxt = self.nxt.pop()
        return self


class FreeVars(ControlValues):
    pass


class ConstrainedVars(ControlValues):
    def __init__(self, control_values):
        sum_of_product = (tuple(zip(*control_values)),)
        super().__init__(sum_of_product)
