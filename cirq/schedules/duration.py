"""A typed time delta that supports picosecond accuracy."""

from typing import Union


class Duration:
    """A time delta that supports picosecond accuracy."""

    def __init__(self,
                 *positional_args,
                 picos: Union[int, float] = 0,
                 nanos: Union[int, float] = 0):
        """Initializes a Duration with a time specified in ns and/or ps.

        If both picos and nanos are specified, their contributions are added.

        Args:
            picos: A number of picoseconds to add to the time delta.
            nanos: A number of nanoseconds to add to the time delta.
        """

        assert not positional_args

        if picos and nanos:
            self._picos = picos + nanos * 1000
        else:
            # Try to preserve type information.
            self._picos = nanos * 1000 if nanos else picos

    def total_picos(self) -> int:
        """Returns the number of picoseconds that the duration spans."""
        return self._picos

    def total_nanos(self) -> float:
        """Returns the number of nanoseconds that the duration spans."""
        return self._picos / 1000.0

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return Duration(picos=self._picos + other._picos)

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return Duration(picos=self._picos - other._picos)

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Duration(picos=self._picos * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
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
        return 'Duration(picos={})'.format(repr(self._picos))
