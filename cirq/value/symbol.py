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

import json
import re

from cirq.value.value_equality import value_equality

_identifier_pattern = '[a-zA-Z_][a-zA-Z0-9_]*$'


def _is_valid_identifier(text):
    return re.match(_identifier_pattern, text)


def _encode(text):
    return json.JSONEncoder().encode(text)


@value_equality
class Symbol:
    """A constant plus the runtime value of a parameter with a given key.

    Attributes:
        name: The non-empty name of a parameter to lookup at runtime and add
            to the constant offset.
    """

    def __init__(self, name: str) -> None:
        """Initializes a Symbol with the given name.

        Args:
            name: The name of a parameter.
        """
        self.name = name

    def __str__(self):
        return (self.name
                if _is_valid_identifier(self.name)
                else 'Symbol({})'.format(_encode(self.name)))

    def __repr__(self):
        return 'cirq.Symbol({!r})'.format(self.name)

    def _value_equality_values_(self):
        return self.name

    def __mul__(self, other):
        return self if other == 1 else NotImplemented

    def __rmul__(self, other):
        return self if other == 1 else NotImplemented

    def __add__(self, other):
        return self if other == 0 else NotImplemented

    def __radd__(self, other):
        return self if other == 0 else NotImplemented
