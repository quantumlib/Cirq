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
from typing import Iterator, List, Sequence, Tuple

import abc
import collections

from cirq.study import resolver


Params = Tuple[Tuple[str, float], ...]


def _check_duplicate_keys(sweeps):
    keys = set()
    for sweep in sweeps:
        if any(key in keys for key in sweep.keys):
            raise ValueError('duplicate keys')
        keys.update(sweep.keys)


class Sweep(metaclass=abc.ABCMeta):
    """A sweep is an iterator over ParamResolvers.

    A ParamResolver assigns values to Symbols. For sweeps, each ParamResolver
    must specify the same Symbols that are assigned.  So a sweep is a way to
    iterate over a set of different values for a fixed set of Symbols. This is
    useful for a circuit, where there are a fixed set of Symbols, and you want
    to iterate over an assignment of all values to all symbols.

    For example, a sweep can explicitly assign a set of equally spaced points
    between two endpoints using a Linspace,
        sweep = Linspace("angle", start=0.0, end=2.0, length=10)
    This can then be used with a circuit that has an 'angle' Symbol to
    run simulations multiple simulations, one for each of the values in the
    sweep
        result = simulator.run_sweep(program=circuit, params=sweep)

    Sweeps support Cartesian and Zip products using the '*' and '+' operators,
    see the Product and Zip documentation.
    """

    def __mul__(self, other: 'Sweep') -> 'Sweep':
        factors = []  # type: List[Sweep]
        if isinstance(self, Product):
            factors.extend(self.factors)
        else:
            factors.append(self)
        if isinstance(other, Product):
            factors.extend(other.factors)
        elif isinstance(other, Sweep):
            factors.append(other)
        else:
            raise TypeError('cannot multiply sweep and {}'.format(type(other)))
        return Product(*factors)

    def __add__(self, other: 'Sweep') -> 'Sweep':
        sweeps = []  # type: List[Sweep]
        if isinstance(self, Zip):
            sweeps.extend(self.sweeps)
        else:
            sweeps.append(self)
        if isinstance(other, Zip):
            sweeps.extend(other.sweeps)
        elif isinstance(other, Sweep):
            sweeps.append(other)
        else:
            raise TypeError('cannot add sweep and {}'.format(type(other)))
        return Zip(*sweeps)

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self == other

    @abc.abstractproperty
    def keys(self) -> List[str]:
        """The keys for the all of the Symbols that are resolved."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[resolver.ParamResolver]:
        for params in self.param_tuples():
            yield resolver.ParamResolver(collections.OrderedDict(params))

    @abc.abstractmethod
    def param_tuples(self) -> Iterator[Params]:
        """An iterator over (key, value) pairs assigning Symbol key to value."""
        pass


class _Unit(Sweep):
    """A sweep with a single element that assigns no parameter values.

    This is useful as a base sweep, instead of special casing None.
    """

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return True

    @property
    def keys(self) -> List[str]:
        return []

    def __len__(self) -> int:
        return 1

    def param_tuples(self) -> Iterator[Params]:
        yield ()

    def __repr__(self):
        return 'cirq.UnitSweep'


UnitSweep = _Unit()  # singleton instance


class Product(Sweep):
    """Cartesian product of one or more sweeps.

    If one sweep assigns 'a' to the values 0, 1, 2, and the second sweep
    assigns 'b' to the values 2, 3, then the product is a sweep that
    assigns the tuple ('a','b') to all possible combinations of these
    assignments: (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3).
    """

    def __init__(self, *factors: Sweep) -> None:
        _check_duplicate_keys(factors)
        self.factors = factors

    def __eq__(self, other):
        return isinstance(other, Product) and self.factors == other.factors

    def __hash__(self):
        return hash(tuple(self.factors))

    @property
    def keys(self) -> List[str]:
        return sum((factor.keys for factor in self.factors), [])

    def __len__(self) -> int:
        if not self.factors:
            return 0
        length = 1
        for factor in self.factors:
            length *= len(factor)
        return length

    def param_tuples(self) -> Iterator[Params]:
        def _gen(factors):
            if not factors:
                yield ()
            else:
                first, rest = factors[0], factors[1:]
                for first_values in first.param_tuples():
                    for rest_values in _gen(rest):
                        yield first_values + rest_values

        return _gen(self.factors)

    def __repr__(self):
        return 'cirq.study.sweeps.Product({})'.format(', '.join(
            repr(f) for f in self.factors))

    def __str__(self):
        if not self.factors:
            return 'Product()'
        factor_strs = []
        for factor in self.factors:
            factor_str = str(factor)
            if isinstance(factor, Zip):
                factor_str = '(' + factor_str + ')'
            factor_strs.append(factor_str)
        return ' * '.join(factor_strs)


class Zip(Sweep):
    """Zip product (direct sum) of one or more sweeps.

    If one sweep assigns 'a' to values 0, 1, 2, and the second sweep assigns 'b'
    to the values 3, 4, 5, then the zip is a sweep that assigns to the
    tuple ('a', 'b') the pair-wise matched values (0, 3), (1, 4), (2, 5).

    When iterating over a Zip, we iterate the individual sweeps in parallel,
    stopping when the first component sweep stops. For example if one sweep
    assigns 'a' to values 0, 1 and the second sweep assigns 'b' to the values
    3, 4, 5, then the zip is a sweep that assigns to the tuple ('a', 'b') the
    values (0, 3), (1, 4).
    """

    def __init__(self, *sweeps: Sweep) -> None:
        _check_duplicate_keys(sweeps)
        self.sweeps = sweeps

    def __eq__(self, other):
        return isinstance(other, Zip) and self.sweeps == other.sweeps

    def __hash__(self):
        return hash(tuple(self.sweeps))

    @property
    def keys(self) -> List[str]:
        return sum((sweep.keys for sweep in self.sweeps), [])

    def __len__(self) -> int:
        if not self.sweeps:
            return 0
        return min(len(sweep) for sweep in self.sweeps)

    def param_tuples(self) -> Iterator[Params]:
        iters = [sweep.param_tuples() for sweep in self.sweeps]
        for values in zip(*iters):
            yield sum(values, ())

    def __repr__(self):
        return 'cirq.study.sweeps.Zip({})'.format(', '.join(
            repr(s) for s in self.sweeps))

    def __str__(self):
        if not self.sweeps:
            return 'Zip()'
        return ' + '.join(str(s) for s in self.sweeps)


class SingleSweep(Sweep):
    """A simple sweep over one parameter with values from an iterator."""

    def __init__(self, key: str) -> None:
        self.key = key

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __hash__(self):
        return hash((self.__class__, self._tuple()))

    @abc.abstractmethod
    def _tuple(self):
        pass

    @property
    def keys(self) -> List[str]:
        return [self.key]

    def param_tuples(self) -> Iterator[Params]:
        for value in self._values():
            yield ((self.key, value),)

    @abc.abstractmethod
    def _values(self) -> Iterator[float]:
        pass


class Points(SingleSweep):
    """A simple sweep with explicitly supplied values."""

    def __init__(self, key: str, points: Sequence[float]) -> None:
        super(Points, self).__init__(key)
        self.points = points

    def _tuple(self):
        return self.key, tuple(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def _values(self) -> Iterator[float]:
        return iter(self.points)

    def __repr__(self):
        return 'cirq.Points({!r}, {!r})'.format(self.key, self.points)


class Linspace(SingleSweep):
    """A simple sweep over linearly-spaced values."""

    def __init__(
        self, key: str,
        start: float,
        stop: float,
        length: int) -> None:
        """Creates a linear-spaced sweep for a given key.

        For the given args, assigns to the list of values
            start, start + (stop - start) / (length - 1), ..., stop
        """
        super(Linspace, self).__init__(key)
        self.start = start
        self.stop = stop
        self.length = length

    def _tuple(self):
        return (self.key, self.start, self.stop, self.length)

    def __len__(self) -> int:
        return self.length

    def _values(self) -> Iterator[float]:
        if self.length == 1:
            yield self.start
        else:
            for i in range(self.length):
                p = i / (self.length - 1)
                yield self.start * (1 - p) + self.stop * p

    def __repr__(self):
        return 'cirq.Linspace({!r}, start={!r}, stop={!r}, length={!r})'.format(
                self.key, self.start, self.stop, self.length)
