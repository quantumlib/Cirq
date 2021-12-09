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
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

import abc
import dataclasses
import sympy

from cirq.value import measurement_key

if TYPE_CHECKING:
    import cirq


class Condition(abc.ABC):
    @property
    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the control keys."""

    @abc.abstractmethod
    def with_keys(self, keys: Tuple['cirq.MeasurementKey', ...]):
        """Replaces the control keys."""

    @abc.abstractmethod
    def resolve(self, measurements: Dict[str, List[int]]) -> bool:
        """Resolves the condition based on the measurements."""


@dataclasses.dataclass(frozen=True)
class KeyCondition(Condition):
    key: 'cirq.MeasurementKey'

    @property
    def keys(self):
        return (self.key,)

    def with_keys(self, keys: Tuple['cirq.MeasurementKey', ...]):
        assert len(keys) == 1
        return KeyCondition(keys[0])

    def __str__(self):
        return str(self.key)

    def resolve(self, measurements: Dict[str, List[int]]) -> bool:
        key = str(self.key)
        if key not in measurements:
            raise ValueError(f'Measurement key {key} missing when testing classical control')
        return any(measurements[key])


@dataclasses.dataclass(frozen=True)
class SympyCondition(Condition):
    expr: sympy.Expr
    control_keys: Tuple['cirq.MeasurementKey', ...]

    @property
    def keys(self):
        return self.control_keys

    def with_keys(self, keys: Tuple['cirq.MeasurementKey', ...]):
        assert len(keys) == len(self.control_keys)
        return dataclasses.replace(self, control_keys=keys)

    def __str__(self):
        return f'({self.expr}, {self.control_keys})'

    def resolve(self, measurements: Dict[str, List[int]]) -> bool:
        missing = [str(k) for k in self.keys if str(k) not in measurements]
        if missing:
            raise ValueError(f'Measurement keys {missing} missing when testing classical control')
        replacements = {f'x{i}': measurements[str(k)][0] for i, k in enumerate(self.keys)}
        return bool(self.expr.subs(replacements))


def parse_sympy_condition(s: str) -> Optional['cirq.SympyCondition']:
    in_key = False
    key_count = 0
    s_out = ''
    key_name = ''
    keys = []
    for c in s:
        if not in_key:
            if c == '{':
                in_key = True
            else:
                s_out += c
        else:
            if c == '}':
                symbol_name = f'x{key_count}'
                s_out += symbol_name
                keys.append(measurement_key.MeasurementKey.parse_serialized(key_name))
                key_name = ''
                key_count += 1
                in_key = False
            else:
                key_name += c
    expr = sympy.sympify(s_out)
    if len(expr.free_symbols) != len(keys):
        return None
    return SympyCondition(expr, tuple(keys))


def parse_condition(s: str) -> 'cirq.Condition':
    c = parse_sympy_condition(s) or measurement_key.MeasurementKey.parse_serialized(s)
    if c is None:
        raise ValueError(f"'{s}' is not a valid condition")
    return c
