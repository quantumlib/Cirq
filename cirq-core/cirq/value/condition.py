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
import dataclasses
from typing import Dict, Mapping, Sequence, Tuple, TYPE_CHECKING

import sympy

from cirq._compat import proper_repr
from cirq.protocols import json_serialization, measurement_key_protocol as mkp
from cirq.value import digits, measurement_key

if TYPE_CHECKING:
    import cirq


class Condition(abc.ABC):
    """A classical control condition that can gate an operation."""

    @property
    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the control keys."""

    @abc.abstractmethod
    def with_keys(self, *keys: 'cirq.MeasurementKey'):
        """Replaces the control keys."""

    @abc.abstractmethod
    def resolve(self, measurements: Mapping[str, Sequence[int]]) -> bool:
        """Resolves the condition based on the measurements."""

    @property
    @abc.abstractmethod
    def qasm(self):
        """Returns the qasm of this condition."""

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'Condition':
        keys = [mkp.with_measurement_key_mapping(k, key_map) for k in self.keys]
        return self.with_keys(*keys)

    def _with_key_path_prefix_(self, path: Tuple[str, ...]) -> 'Condition':
        keys = [mkp.with_key_path_prefix(k, path) for k in self.keys]
        return self.with_keys(*keys)


@dataclasses.dataclass(frozen=True)
class KeyCondition(Condition):
    """A classical control condition based on a single measurement key.

    This condition resolves to True iff the measurement key is non-zero at the
    time of resolution.
    """

    key: 'cirq.MeasurementKey'

    @property
    def keys(self):
        return (self.key,)

    def with_keys(self, *keys: 'cirq.MeasurementKey'):
        if len(keys) != 1:
            raise ValueError(f'Cannot apply multiple keys to a KeyCondition')
        return KeyCondition(keys[0])

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return f'cirq.KeyCondition({self.key!r})'

    def resolve(self, measurements: Mapping[str, Sequence[int]]) -> bool:
        key = str(self.key)
        if key not in measurements:
            raise ValueError(f'Measurement key {key} missing when testing classical control')
        return any(measurements[key])

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @property
    def qasm(self):
        return f'm_{self.key}!=0'


@dataclasses.dataclass(frozen=True)
class SympyCondition(Condition):
    """A classical control condition based on a sympy expression.

    This condition resolves to True iff the sympy expression resolves to a
    truthy value (i.e. `bool(x) == True`) when the measurement keys are
    substituted in as the free variables.

    To account for the fact that measurement key strings can contain characters
    not allowed in sympy variables, we use x0..xN for the free variables and
    substitute the measurement keys in by order. For instance a condition with
    `expr='x0 > x1', keys=['0:A', '0:B']` would resolve to True iff key `0:A`
    is greater than key `0:B`.

    The `cirq.parse_sympy_condition` function automates setting this up
    correctly. To create the above expression, one would call
    `cirq.parse_sympy_condition('{0:A} > {0:B}')`.
    """

    expr: sympy.Expr
    control_keys: Tuple['cirq.MeasurementKey', ...]

    @property
    def keys(self):
        return self.control_keys

    def with_keys(self, *keys: 'cirq.MeasurementKey'):
        if len(keys) != len(self.control_keys):
            raise ValueError(f'Wrong number of keys applied to this condition.')
        return dataclasses.replace(self, control_keys=keys)

    def __str__(self):
        replacements = {f'x{i}': f'{{{str(key)}}}' for i, key in enumerate(self.control_keys)}

        class CustomCodePrinter(sympy.printing.StrPrinter):
            def _print_Symbol(self, expr):
                return replacements[expr.name]

        return CustomCodePrinter().doprint(self.expr)

    def __repr__(self):
        return f'cirq.SympyCondition({proper_repr(self.expr)}, {self.keys!r})'

    def resolve(self, measurements: Mapping[str, Sequence[int]]) -> bool:
        missing = [str(k) for k in self.keys if str(k) not in measurements]
        if missing:
            raise ValueError(f'Measurement keys {missing} missing when testing classical control')

        def value(k):
            return digits.big_endian_bits_to_int(measurements[str(k)])

        replacements = {f'x{i}': value(k) for i, k in enumerate(self.keys)}
        return bool(self.expr.subs(replacements))

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @property
    def qasm(self):
        raise NotImplementedError()


def parse_sympy_condition(s: str) -> 'cirq.SympyCondition':
    """Parses a string into a `cirq.SympyCondition`.

    The measurement keys in a sympy condition string must be wrapped in curly
    braces to denote them. For example, to create an expression that checks if
    measurement A was greater than measurement B, the proper syntax is
    `cirq.parse_sympy_condition('{A} > {B}')`.

    A backslash can be used to treat the subsequent character as a literal
    within the key name, in case braces or backslashes appear in the key name.
    """
    in_key = False
    key_count = 0
    s_out = ''
    key_name = ''
    keys = []
    i = 0
    while i < len(s):
        c = s[i]
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
                if c == '\\':
                    i += 1
                    if i == len(s):
                        raise ValueError(f"'{s}' is not a valid sympy condition")
                    c = s[i]
                key_name += c
        i += 1
    if in_key:
        raise ValueError(f"'{s}' is not a valid sympy condition")
    expr = sympy.sympify(s_out)
    if len(expr.free_symbols) != len(keys):
        raise ValueError(f"'{s}' is not a valid sympy condition")
    return SympyCondition(expr, tuple(keys))


def parse_condition(s: str) -> 'cirq.Condition':
    """Parses a string into a `Condition`."""
    return (
        parse_sympy_condition(s)
        if '{' in s
        else KeyCondition(measurement_key.MeasurementKey.parse_serialized(s))
    )
