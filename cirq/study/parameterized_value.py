# Copyright 2018 Google LLC
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


Parameterizable = Union['ParameterizedValue', float, str]


class ParameterizedValue:
    """A constant plus the runtime value of a parameter with a given key.

    Attributes:
        val: The constant offset.
        key: The non-empty name of a parameter to lookup at runtime and add
            to the constant offset.
    """

    def __new__(cls, key: str = '', val: float = 0):
        """Constructs a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. If this is the empty string, then no
                parameter will be used.

        Returns:
            Just val if key is empty, otherwise a new ParameterizedValue.
        """
        if key == '':
            return val
        return super().__new__(cls)

    def __init__(self, key: str = '', val: float = 0):
        """Initializes a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. Because of the implementation of new,
                this will never be the empty string.
        """
        self.val = val
        self.key = key

    def __str__(self):
        if self.key == '':
            return repr(self.val)
        key_rep = (self.key
                   if self.key.isalpha()
                   else 'param({})'.format(repr(self.key)))
        if self.val == 0:
            return key_rep
        return '{}+{}'.format(repr(self.val), key_rep)

    def __repr__(self):
        return 'ParameterizedValue({}, {})'.format(repr(self.val),
                                                   repr(self.key))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key and self.val == other.val

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ParameterizedValue, self.val, self.key))

    def __add__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val + other)

    def __radd__(self, other: float) -> 'ParameterizedValue':
        return self.__add__(other)

    def __sub__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val - other)

    @staticmethod
    def val_of(val: Parameterizable):
        if isinstance(val, ParameterizedValue):
            return float(val.val)
        elif isinstance(val, str):
            return 0.0
        return float(val)

    @staticmethod
    def key_of(val: Parameterizable):
        if isinstance(val, ParameterizedValue):
            return val.key
        elif isinstance(val, str):
            return val
        return ''
