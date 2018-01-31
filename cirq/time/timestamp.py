"""A typed location in time that supports picosecond accuracy."""

from typing import Union

from cirq.time.duration import Duration


class Timestamp:
    """A location in time with picosecond accuracy.

    Supports affine operations against Duration."""

    def __init__(self,
                 *positional_args,
                 picos: Union[int, float] = 0,
                 nanos: Union[int, float] = 0):
        """Initializes a Timestamp with a time specified in ns and/or ps.

        The time is relative to some unspecified "time zero". If both picos and
        nanos are specified, their contributions away from zero are added.

        Args:
            picos: How many picoseconds away from time zero?
            nanos: How many nanoseconds away from time zero?
        """

        assert not positional_args

        if picos and nanos:
            self._picos = picos + nanos * 1000
        else:
            # Try to preserve type information.
            self._picos = nanos * 1000 if nanos else picos

    def raw_picos(self) -> int:
        """The timestamp's location in picoseconds from arbitrary time zero."""
        return self._picos

    def __add__(self, other):
        if not isinstance(other, Duration):
            return NotImplemented
        return Timestamp(picos=self._picos + other.total_picos())

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Duration):
            return Timestamp(picos=self._picos - other.total_picos())
        if isinstance(other, type(self)):
            return Duration(picos=self._picos - other._picos)
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
        return hash((Timestamp, self._picos))

    def __str__(self):
        return 't={}'.format(self._picos)

    def __repr__(self):
        return 'Timestamp(picos={})'.format(repr(self._picos))
