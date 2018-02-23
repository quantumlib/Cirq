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


class ParameterizedValue:
    """A constant plus the runtime value of a parameter with a given key.

    Attributes:
        val: The constant offset.
        key: The non-empty name of a parameter to lookup at runtime and add
            to the constant offset.
    """

    def __new__(cls, key: str = '', val: float = 0, factor: float = 1):
        """Constructs a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. If this is the empty string, then no
                parameter will be used.
            factor: Scales the parameter's value (but not the offset val).

        Returns:
            Just val if key is empty, otherwise a new ParameterizedValue.
        """
        if key == '' or factor == 0:
            return val
        return super().__new__(cls)

    def __init__(self, key: str = '', val: float = 0, factor: float = 1):
        """Initializes a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. Because of the implementation of new,
                this will never be the empty string.
            factor: Scales the parameter's value (but not the offset val).
        """
        self.val = val
        self.key = key
        self.factor = 1 if key == '' else factor

    def __str__(self):
        if self.key == '':
            return repr(self.val)
        key_rep = (self.key
                   if self.key.isalpha()
                   else 'param({})'.format(repr(self.key)))
        if self.val == 0 and self.factor == 1:
            return key_rep
        if self.val == 0:
            return '{}*{}'.format(key_rep, repr(self.factor))
        if self.factor == 1:
            return '{}+{}'.format(repr(self.val), key_rep)
        return '{} + {}*{}'.format(repr(self.val),
                                   key_rep,
                                   repr(self.factor))

    def __repr__(self):
        return 'ParameterizedValue({}, {}, {})'.format(repr(self.val),
                                                       repr(self.key),
                                                       repr(self.factor))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.key == other.key and
                self.val == other.val and
                self.factor == other.factor)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ParameterizedValue, self.val, self.key, self.factor))

    def __add__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val + other, self.factor)

    def __radd__(self, other: float) -> 'ParameterizedValue':
        return self.__add__(other)

    def __sub__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val - other, self.factor)

    def __mul__(self, other: float) -> 'ParameterizedValue':
        return ParameterizedValue(self.key,
                                  self.val * other,
                                  self.factor * other)

    def __rmul__(self, other: float) -> 'ParameterizedValue':
        return self.__mul__(other)

    def __neg__(self) -> 'ParameterizedValue':
        return self.__mul__(-1)

    def __truediv__(self, other: float) -> 'ParameterizedValue':
        return ParameterizedValue(self.key,
                                  self.val / other,
                                  self.factor / other)

    @staticmethod
    def val_of(val: Union['ParameterizedValue', float]) -> float:
        if isinstance(val, ParameterizedValue):
            return float(val.val)
        return float(val)

    @staticmethod
    def factor_of(val: Union['ParameterizedValue', float]) -> float:
        if isinstance(val, ParameterizedValue):
            return val.factor
        return 0

    @staticmethod
    def key_of(val: Union['ParameterizedValue', float]) -> str:
        if isinstance(val, ParameterizedValue):
            return val.key
        return ''
