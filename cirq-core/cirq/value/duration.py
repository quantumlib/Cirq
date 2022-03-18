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

from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union
import datetime

import sympy

from cirq import protocols
from cirq._compat import proper_repr
from cirq._doc import document

if TYPE_CHECKING:
    import cirq

DURATION_LIKE = Union[None, datetime.timedelta, 'cirq.Duration']
document(
    DURATION_LIKE,  # type: ignore
    """A `cirq.Duration` or value that can trivially converted to one.

    A `datetime.timedelta` is a `cirq.DURATION_LIKE`. It is converted while
    preserving its duration.

    `None` is a `cirq.DURATION_LIKE` that converts into a zero-length duration.

    Note that 0 is a `DURATION_LIKE`, despite the fact that `int` is not listed,
    because 0 is the only integer where the physical unit doesn't matter.
    """,
)


class Duration:
    """A time delta that supports symbols and picosecond accuracy."""

    def __init__(
        self,
        value: DURATION_LIKE = None,
        *,  # Force keyword args.
        picos: Union[int, float, sympy.Basic] = 0,
        nanos: Union[int, float, sympy.Basic] = 0,
        micros: Union[int, float, sympy.Basic] = 0,
        millis: Union[int, float, sympy.Basic] = 0,
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
        if value is not None and value != 0:
            if isinstance(value, datetime.timedelta):
                # timedelta has microsecond resolution.
                micros += int(value / datetime.timedelta(microseconds=1))
            elif isinstance(value, Duration):
                picos += value._picos
            else:
                raise TypeError(f'Not a `cirq.DURATION_LIKE`: {repr(value)}.')

        self._picos: Union[float, int, sympy.Basic] = (
            picos + nanos * 1000 + micros * 1000_000 + millis * 1000_000_000
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._picos)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._picos)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'Duration':
        return Duration(picos=protocols.resolve_parameters(self._picos, resolver, recursive))

    def total_picos(self) -> Union[sympy.Basic, float]:
        """Returns the number of picoseconds that the duration spans."""
        return self._picos

    def total_nanos(self) -> Union[sympy.Basic, float]:
        """Returns the number of nanoseconds that the duration spans."""
        return self._picos / 1000

    def total_micros(self) -> Union[sympy.Basic, float]:
        """Returns the number of microseconds that the duration spans."""
        return self._picos / 1000_000

    def total_millis(self) -> Union[sympy.Basic, float]:
        """Returns the number of milliseconds that the duration spans."""
        return self._picos / 1000_000_000

    def __add__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return Duration(picos=self._picos + other._picos)

    def __radd__(self, other) -> 'Duration':
        return self.__add__(other)

    def __sub__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return Duration(picos=self._picos - other._picos)

    def __rsub__(self, other) -> 'Duration':
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return Duration(picos=other._picos - self._picos)

    def __mul__(self, other) -> 'Duration':
        if not isinstance(other, (int, float, sympy.Basic)):
            return NotImplemented
        return Duration(picos=self._picos * other)

    def __rmul__(self, other) -> 'Duration':
        return self.__mul__(other)

    def __truediv__(self, other) -> Union['Duration', float]:
        if isinstance(other, (int, float, sympy.Basic)):
            return Duration(picos=self._picos / other)

        other_duration = _attempt_duration_like_to_duration(other)
        if other_duration is not None:
            return self._picos / other_duration._picos

        return NotImplemented

    def __eq__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos == other._picos

    def __ne__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos != other._picos

    def __gt__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos > other._picos

    def __lt__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos < other._picos

    def __ge__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos >= other._picos

    def __le__(self, other):
        other = _attempt_duration_like_to_duration(other)
        if other is None:
            return NotImplemented
        return self._picos <= other._picos

    def __bool__(self):
        return bool(self._picos)

    def __hash__(self):
        if isinstance(self._picos, (int, float)) and self._picos % 1000000 == 0:
            return hash(datetime.timedelta(microseconds=self._picos / 1000000))
        return hash((Duration, self._picos))

    def _decompose_into_amount_unit_suffix(self) -> Tuple[int, str, str]:
        if (
            isinstance(self._picos, sympy.Mul)
            and len(self._picos.args) == 2
            and isinstance(self._picos.args[0], (sympy.Integer, sympy.Float))
        ):
            scale = self._picos.args[0]
            rest = self._picos.args[1]
        else:
            scale = self._picos
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
        if self._picos == 0:
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
