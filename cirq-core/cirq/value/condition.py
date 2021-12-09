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

from typing import (
    Tuple,
    TYPE_CHECKING,
)

import sympy

if TYPE_CHECKING:
    import cirq


class Condition:
    def __init__(self, expr: sympy.Expr, keys: Tuple['cirq.MeasurementKey', ...]):
        self._expr = expr
        self._keys = keys

    @property
    def keys(self):
        return self._keys

    @property
    def expr(self):
        return self._expr

    def with_keys(self, keys: Tuple['cirq.MeasurementKey', ...]):
        assert len(keys) == len(self._keys)
        return Condition(self._expr, keys)

    def __eq__(self, x):
        return isinstance(x, Condition) and self._keys == x._keys and self._expr == x._expr

    def __hash__(self):
        return hash(self._keys) ^ hash(self._expr)

    def __str__(self):
        if self._expr == sympy.symbols('x0') and len(self._keys) == 1:
            return str(self._keys[0])
        return f'({self._expr}, {self._keys})'
