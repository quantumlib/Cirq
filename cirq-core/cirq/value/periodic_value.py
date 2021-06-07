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

from typing import AbstractSet, Any, TYPE_CHECKING, Union

import sympy

from cirq._compat import proper_repr


if TYPE_CHECKING:
    import cirq


class PeriodicValue:
    """Wrapper for periodic numerical values.

    Wrapper for periodic numerical types which implements `__eq__`, `__ne__`,
    `__hash__` and `_approx_eq_` so that values which are in the same
    equivalence class are treated as equal.

    Internally the `value` passed to `__init__` is normalized to the interval
    [0, `period`) and stored as that. Specialized version of `_approx_eq_` is
    provided to cover values which end up at the opposite edges of this
    interval.
    """

    def __init__(self, value: Union[int, float], period: Union[int, float]):
        """Initializes the equivalence class.

        Args:
            value: numerical value to wrap.
            period: periodicity of the numerical value.
        """
        self.value = value % period
        self.period = period

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.value, self.period) == (other.value, other.period)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((type(self), self.value, self.period))

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        """Implementation of `SupportsApproximateEquality` protocol."""
        # HACK: Avoids circular dependencies.
        from cirq.protocols import approx_eq

        if not isinstance(other, type(self)):
            return NotImplemented

        # self.value = value % period in __init__() creates a Mod
        if isinstance(other.value, sympy.Mod):
            return self.value == other.value
        # Periods must be exactly equal to avoid drift of normalized value when
        # original value increases.
        if self.period != other.period:
            return False

        low = min(self.value, other.value)
        high = max(self.value, other.value)

        # Shift lower value outside of normalization interval in case low and
        # high values are at the opposite borders of normalization interval.
        if high - low > self.period / 2:
            low += self.period

        return approx_eq(low, high, atol=atol)

    def __repr__(self) -> str:
        v = proper_repr(self.value)
        p = proper_repr(self.period)
        return f'cirq.PeriodicValue({v}, {p})'

    def _is_parameterized_(self) -> bool:
        # HACK: Avoids circular dependencies.
        from cirq.protocols import is_parameterized

        return is_parameterized(self.value) or is_parameterized(self.period)

    def _parameter_names_(self) -> AbstractSet[str]:
        # HACK: Avoids circular dependencies.
        from cirq.protocols import parameter_names

        return parameter_names(self.value) | parameter_names(self.period)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'PeriodicValue':
        # HACK: Avoids circular dependencies.
        from cirq.protocols import resolve_parameters

        return PeriodicValue(
            value=resolve_parameters(self.value, resolver, recursive),
            period=resolve_parameters(self.period, resolver, recursive),
        )
