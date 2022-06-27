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
from typing import Mapping, Tuple, TYPE_CHECKING, FrozenSet

import sympy

from cirq._compat import proper_repr
from cirq.protocols import json_serialization, measurement_key_protocol as mkp
from cirq.value import measurement_key

if TYPE_CHECKING:
    import cirq


class Condition(abc.ABC):
    """A classical control condition that can gate an operation."""

    @property
    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the control keys."""

    @abc.abstractmethod
    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        """Replaces the control keys."""

    @abc.abstractmethod
    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        """Resolves the condition based on the measurements."""

    @property
    @abc.abstractmethod
    def qasm(self):
        """Returns the qasm of this condition."""

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> 'cirq.Condition':
        condition = self
        for k in self.keys:
            condition = condition.replace_key(k, mkp.with_measurement_key_mapping(k, key_map))
        return condition

    def _with_key_path_prefix_(self, path: Tuple[str, ...]) -> 'cirq.Condition':
        condition = self
        for k in self.keys:
            condition = condition.replace_key(k, mkp.with_key_path_prefix(k, path))
        return condition

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ) -> 'cirq.Condition':
        condition = self
        for key in self.keys:
            for i in range(len(path) + 1):
                back_path = path[: len(path) - i]
                new_key = key.with_key_path_prefix(*back_path)
                if new_key in bindable_keys:
                    condition = condition.replace_key(key, new_key)
                    break
        return condition


@dataclasses.dataclass(frozen=True)
class KeyCondition(Condition):
    """A classical control condition based on a single measurement key.

    This condition resolves to True iff the measurement key is non-zero at the
    time of resolution.
    """

    key: 'cirq.MeasurementKey'
    index: int = -1

    @property
    def keys(self):
        return (self.key,)

    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        return KeyCondition(replacement) if self.key == current else self

    def __str__(self):
        return str(self.key) if self.index == -1 else f'{self.key}[{self.index}]'

    def __repr__(self):
        if self.index != -1:
            return f'cirq.KeyCondition({self.key!r}, {self.index})'
        return f'cirq.KeyCondition({self.key!r})'

    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        if self.key not in classical_data.keys():
            raise ValueError(f'Measurement key {self.key} missing when testing classical control')
        return classical_data.get_int(self.key, self.index) != 0

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @classmethod
    def _from_json_dict_(cls, key, **kwargs):
        return cls(key=key)

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')


@dataclasses.dataclass(frozen=True)
class SympyCondition(Condition):
    """A classical control condition based on a sympy expression.

    This condition resolves to True iff the sympy expression resolves to a
    truthy value (i.e. `bool(x) == True`) when the measurement keys are
    substituted in as the free variables.
    """

    expr: sympy.Basic

    @property
    def keys(self):
        return tuple(
            measurement_key.MeasurementKey.parse_serialized(symbol.name)
            for symbol in self.expr.free_symbols
        )

    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        return SympyCondition(self.expr.subs({str(current): sympy.Symbol(str(replacement))}))

    def __str__(self):
        return str(self.expr)

    def __repr__(self):
        return f'cirq.SympyCondition({proper_repr(self.expr)})'

    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        missing = [str(k) for k in self.keys if k not in classical_data.keys()]
        if missing:
            raise ValueError(f'Measurement keys {missing} missing when testing classical control')

        replacements = {str(k): classical_data.get_int(k) for k in self.keys}
        return bool(self.expr.subs(replacements))

    def _json_dict_(self):
        return json_serialization.dataclass_json_dict(self)

    @classmethod
    def _from_json_dict_(cls, expr, **kwargs):
        return cls(expr=expr)

    @property
    def qasm(self):
        if isinstance(self.expr, sympy.Equality):
            if isinstance(self.expr.lhs, sympy.Symbol) and isinstance(self.expr.rhs, sympy.Integer):
                # Measurements get prepended with "m_", so the condition needs to be too.
                return f'm_{self.expr.lhs}=={self.expr.rhs}'
        raise ValueError('QASM is defined only for SympyConditions of type key == constant.')
