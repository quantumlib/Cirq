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
"""A typed time delta that supports picosecond accuracy."""

from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime

import sympy
import numpy as np

from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document

if TYPE_CHECKING:
    import cirq

DURATION_LIKE = Union[None, datetime.timedelta, 'cirq.Duration']
document(
    DURATION_LIKE,
    """A `cirq.Duration` or value that can trivially converted to one.

    A `datetime.timedelta` is a `cirq.DURATION_LIKE`. It is converted while
    preserving its duration.

    `None` is a `cirq.DURATION_LIKE` that converts into a zero-length duration.

    Note that 0 is a `DURATION_LIKE`, despite the fact that `int` is not listed,
    because 0 is the only integer where the physical unit doesn't matter.
    """,
)


_NUMERIC_INPUT_TYPE = Union[int, float, sympy.Expr, np.number]
_NUMERIC_OUTPUT_TYPE = Union[int, float, sympy.Expr]


class Duration:
    """A time delta that supports symbols and picosecond accuracy."""

    def __init__(
        self,
        value: DURATION_LIKE = None,
        *,  # Force keyword args.
        picos: _NUMERIC_INPUT_TYPE = 0,
        nanos: _NUMERIC_INPUT_TYPE = 0,
        micros: _NUMERIC_INPUT_TYPE = 0,
        millis: _NUMERIC_INPUT_TYPE = 0,
    ) -> None:
        """Initializes a Duration with a time specified in some unit.

        If multiple arguments are specified, their contributions are added.

        Args:
            value: A value with a pre-specified time unit. Currently only
                supports 0 and `datetime.timedelta` instances.
            picos: A number of picoseconds to add to the time delta.
            nanos: A number of nanoseconds to add to the time delta.
            micros: A number of microseconds to add to the time delta.
            millis: A number of milliseconds to add to the time delta.

        Raises:
            TypeError: If the given value is not of a `cirq.DURATION_LIKE` type.

        Examples:
            >>> print(cirq.Duration(nanos=100))
            100 ns
            >>> print(cirq.Duration(micros=1.5 * sympy.Symbol('t')))
            (1500.0*t) ns
        """
        self._time_vals: List[_NUMERIC_INPUT_TYPE] = [0, 0, 0, 0]
        self._multipliers = [1, 1000, 1000_000, 1000_000_000]
        if value is not None and value != 0:
            if isinstance(value, datetime.timedelta):
                # timedelta has microsecond resolution.
                self._time_vals[2] = int(value / datetime.timedelta(microseconds=1))
            elif isinstance(value, Duration):
                self._time_vals = value._time_vals
            else:
                raise TypeError(f'Not a `cirq.DURATION_LIKE`: {repr(value)}.')
        input_vals = [picos, nanos, micros, millis]
        self._time_vals = _add_time_vals(self._time_vals, input_vals)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._time_vals)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._time_vals)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'Duration':
        return _duration_from_time_vals(
            protocols.resolve_parameters(self._time_vals, resolver, recursive)
        )

    @cached_method
    def total_picos(self) -> _NUMERIC_OUTPUT_TYPE:
        """Returns the number of picoseconds that the duration spans."""
        val = sum(a * b for a, b in zip(self._time_vals, self._multipliers))
        return float(val) if isinstance(val, np.number) else val

    def total_nanos(self) -> _NUMERIC_OUTPUT_TYPE:
        """Returns the number of nanoseconds that the duration spans."""
        return self.total_picos() / 1000

    def total_micros(self) -> _NUMERIC_OUTPUT_TYPE:
        """Returns the number of microseconds that the duration spans."""
        return self.total_picos() / 1000_000

    def total_millis(self) -> _NUMERIC_OUTPUT_TYPE:
        """Returns the number of milliseconds that the duration spans."""
        return self.total_picos() / 1000_000_000

    def __add__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return _duration_from_time_vals(_add_time_vals(self._time_vals, other._time_vals))

    def __radd__(self, other) -> 'Duration':
        return self.__add__(other)

    def __sub__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return _duration_from_time_vals(
            _add_time_vals(self._time_vals, [-x for x in other._time_vals])
        )

    def __rsub__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return _duration_from_time_vals(
            _add_time_vals(other._time_vals, [-x for x in self._time_vals])
        )

    def __mul__(self, other) -> 'Duration':
        if not isinstance(other, (int, float, sympy.Expr)):
            return NotImplemented
        if other == 0:
            return _duration_from_time_vals([0] * 4)
        return _duration_from_time_vals([x * other for x in self._time_vals])

    def __rmul__(self, other) -> 'Duration':
        return self.__mul__(other)

    def __truediv__(self, other) -> Union['Duration', float]:
        if isinstance(other, (int, float, sympy.Expr)):
            new_time_vals = [x / other for x in self._time_vals]
            return _duration_from_time_vals(new_time_vals)

        other_duration = _attempt_duration_like_to_duration(other)
        if other_duration is not None:
            return self.total_picos() / other_duration.total_picos()

        return NotImplemented

    def __eq__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() == other.total_picos()

    def __ne__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() != other.total_picos()

    def __gt__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() > other.total_picos()

    def __lt__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() < other.total_picos()

    def __ge__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() >= other.total_picos()

    def __le__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self.total_picos() <= other.total_picos()

    def __bool__(self):
        return bool(self.total_picos())

    def __hash__(self):
        if isinstance(self.total_picos(), (int, float)) and self.total_picos() % 1000000 == 0:
            return hash(datetime.timedelta(microseconds=self.total_picos() / 1000000))
        return hash((Duration, self.total_picos()))

    def _decompose_into_amount_unit_suffix(self) -> Tuple[int, str, str]:
        picos = self.total_picos()
        if (
            isinstance(picos, sympy.Mul)
            and len(picos.args) == 2
            and isinstance(picos.args[0], (sympy.Integer, sympy.Float))
        ):
            scale = picos.args[0]
            rest = picos.args[1]
        else:
            scale = picos
            rest = 1

        if scale % 1000_000_000 == 0:
            amount = scale / 1000_000_000
            unit = 'millis'
            suffix = 'ms'
        elif scale % 1000_000 == 0:
            amount = scale / 1000_000
            unit = 'micros'
            suffix = 'us'
        elif scale % 1000 == 0:
            amount = scale / 1000
            unit = 'nanos'
            suffix = 'ns'
        else:
            amount = scale
            unit = 'picos'
            suffix = 'ps'

        if isinstance(scale, int):
            amount = int(amount)

        return amount * rest, unit, suffix

    def __str__(self) -> str:
        if self.total_picos() == 0:
            return 'Duration(0)'
        amount, _, suffix = self._decompose_into_amount_unit_suffix()
        if not isinstance(amount, (int, float, sympy.Symbol)):
            amount = f'({amount})'
        return f'{amount} {suffix}'

    def __repr__(self) -> str:
        amount, unit, _ = self._decompose_into_amount_unit_suffix()
        return f'cirq.Duration({unit}={proper_repr(amount)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {'picos': self.total_picos()}


def _attempt_duration_like_to_duration(value: Any) -> Optional[Duration]:
    if isinstance(value, Duration):
        return value
    if isinstance(value, datetime.timedelta):
        return Duration(value)
    if isinstance(value, (int, float)) and value == 0:
        return Duration()
    return None


def _add_time_vals(
    val1: List[_NUMERIC_INPUT_TYPE], val2: List[_NUMERIC_INPUT_TYPE]
) -> List[_NUMERIC_INPUT_TYPE]:
    ret: List[_NUMERIC_INPUT_TYPE] = []
    for i in range(4):
        if val1[i] and val2[i]:
            ret.append(val1[i] + val2[i])
        else:
            ret.append(val1[i] or val2[i])
    return ret


def _duration_from_time_vals(time_vals: List[_NUMERIC_INPUT_TYPE]):
    ret = Duration()
    ret._time_vals = time_vals
    return ret
