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

from typing import Union


class Duration:
    """A time delta that supports picosecond accuracy."""

    def __init__(self, *,  # Forces keyword args.
                 picos: Union[int, float] = 0,
                 nanos: Union[int, float] = 0) -> None:
        """Initializes a Duration with a time specified in ns and/or ps.

        If both picos and nanos are specified, their contributions are added.

        Args:
            picos: A number of picoseconds to add to the time delta.
            nanos: A number of nanoseconds to add to the time delta.
        """

        if picos and nanos:
            self._picos = picos + nanos * 1000
        else:
            # Try to preserve type information.
            self._picos = nanos * 1000 if nanos else picos

    def total_picos(self) -> float:
        """Returns the number of picoseconds that the duration spans."""
        return self._picos

    def total_nanos(self) -> float:
        """Returns the number of nanoseconds that the duration spans."""
        return self._picos / 1000.0

    def __add__(self, other) -> 'Duration':
        if not isinstance(other, type(self)):
            return NotImplemented
        return Duration(picos=self._picos + other._picos)

    def __sub__(self, other) -> 'Duration':
        if not isinstance(other, type(self)):
            return NotImplemented
        return Duration(picos=self._picos - other._picos)

    def __mul__(self, other) -> 'Duration':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Duration(picos=self._picos * other)

    def __rmul__(self, other) -> 'Duration':
        return self.__mul__(other)

    def __truediv__(self, other) -> Union['Duration', float]:
        if isinstance(other, (int, float)):
            return Duration(picos=self._picos / other)
        if isinstance(other, type(self)):
            return self._picos / other._picos
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._picos == other._picos

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._picos > other._picos

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._picos < other._picos

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def __hash__(self):
        return hash((Duration, self._picos))

    def __str__(self):
        if self._picos % 1000 == 0:
            return '{}ns'.format(self._picos // 1000)
        return '{}ps'.format(self._picos)

    def __repr__(self):
        return 'cirq.Duration(picos={})'.format(repr(self._picos))
