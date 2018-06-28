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

from typing import Union


class PauliMeta(type):
    def __call__(cls, index: int):
        if not 0 <= index < len(cls._instances):
            raise IndexError('{} index out of range'.format(cls.__name__))
        return cls._instances[index]
    def __init__(cls, name, supers, attrs):
        super().__init__(name, supers, attrs)
        cls._instances = tuple((super(PauliMeta, cls).__call__(i)
                                for i in range(cls.instance_count)))


class Pauli(metaclass=PauliMeta):
    '''Represents the X, Y, or Z axis of the Bloch sphere.'''
    instance_count = 3

    def __init__(self, index: int) -> None:
        self._index = index

    def commutes_with(self, other: 'Pauli') -> bool:
        return self is other

    def third(self, second: 'Pauli') -> 'Pauli':
        return Pauli((-self._index - second._index) % 3)

    def difference(self, second: 'Pauli') -> int:
        return (self._index - second._index + 1) % 3 - 1

    def __add__(self, shift: int) -> 'Pauli':
        return Pauli((self._index + shift) % 3)

    def __sub__(self, other_or_shift: Union['Pauli', int]
                ) -> Union[int, 'Pauli']:
        if isinstance(other_or_shift, int):
            return self + -other_or_shift
        else:
            return self.difference(other_or_shift)

    def __str__(self):
        return 'XYZ'[self._index]

    def __repr__(self):
        return 'PAULI_{!s}'.format(self)


PAULI_X = Pauli(0)
PAULI_Y = Pauli(1)
PAULI_Z = Pauli(2)
