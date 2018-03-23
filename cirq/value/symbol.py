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


class Symbol:
    """A constant plus the runtime value of a parameter with a given key.

    Attributes:
        val: The constant offset.
        name: The non-empty name of a parameter to lookup at runtime and add
            to the constant offset.
    """

    def __init__(self, name: str) -> None:
        """Initializes a Symbol with the given name.

        Args:
            name: The name of a parameter. Because of the implementation of new,
                this will never be the empty string.
        """
        self.name = name

    def __str__(self):
        return (self.name
                if self.name.isalpha()
                else 'Symbol({!r})'.format(self.name))

    def __repr__(self):
        return 'Symbol({!r})'.format(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Symbol, self.name))
