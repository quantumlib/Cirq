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

import cirq

from cirq.value.value_equality import value_equality


@value_equality
class PeriodicEquivalence:
    """Equivalence class wrapper for periodic numerical values.

    Wrapper for periodic numerical types which implements `__eq__`, `__ne__`,
    `__hash__` and `_approx_eq_` so that values which are in the same
    equivalence class are treated as equal.

    Internally the `value` passed to `__init__` is normalized to the interval
    [0, `period`) and stored as is. Specialized version of `_approx_eq_` is
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

    def _value_equality_values_(self):
        """Implementation of `_SupportsValueEquality` protocol."""
        return self.value, self.period

    def _approx_eq_(self, other: 'PeriodicEquivalence', atol: float):
        """Implementation of `SupportsApproximateEquality` protocol."""
        if type(self) != type(other):
            return NotImplemented

        # Periods must be approximately equal.
        if not cirq.approx_eq(self.period, other.period, atol=atol):
            return False

        # Try to match normalized values.
        if cirq.approx_eq(self.value, other.value, atol=atol):
            return True

        # Shift lower value outside of normalization interval in case self and
        # other values are at the opposite borders of normalization interval.
        if self.value < other.value:
            s_val = self.value + self.period
            o_val = other.value
        else:
            s_val = self.value
            o_val = other.value + other.period
        return cirq.approx_eq(s_val, o_val, atol=atol)
