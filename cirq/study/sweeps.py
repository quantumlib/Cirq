import abc
from typing import Iterator, List, Sequence, Tuple

from cirq.study.resolver import ParamResolver


Params = Tuple[Tuple[str, float], ...]


def _check_duplicate_keys(sweeps):
    keys = set()
    for sweep in sweeps:
        if any(key in keys for key in sweep.keys):
            raise ValueError('duplicate keys')
        keys.update(sweep.keys)


class Sweep(metaclass=abc.ABCMeta):
    def __mul__(self, other: 'Sweep') -> 'Sweep':
        factors = []
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
        sweeps = []
        if isinstance(self, Zip):
            sweeps.extend(self.sweeps)
        else:
            sweeps.append(self)
        if isinstance(other, Zip):
            sweeps.extend(other.factors)
        elif isinstance(other, Sweep):
            sweeps.append(other)
        else:
            raise TypeError('cannot add sweep and {}'.format(type(other)))
        return Zip(*sweeps)

    @abc.abstractproperty
    def keys(self) -> List[str]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Params]:
        pass

    def resolvers(self) -> Iterator[ParamResolver]:
        for params in self:
            yield ParamResolver(dict(params))


class _Unit(Sweep):
    """A sweep with a single element that assigns no parameter values."""

    @property
    def keys(self) -> List[str]:
        return []

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[Params]:
        yield ()

    def __repr__(self):
        return 'Unit'


Unit = _Unit()  # singleton instance


class Product(Sweep):
    """Cartesian product of one or more sweeps."""

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

    def __iter__(self) -> Iterator[Params]:
        if not self.factors:
            return

        def _gen(factors):
            if not factors:
                yield ()
            else:
                first, rest = factors[0], factors[1:]
                for first_values in first:
                    for rest_values in _gen(rest):
                        yield first_values + rest_values

        return _gen(self.factors)

    def __repr__(self):
        return 'Product({})'.format(', '.join(repr(f) for f in self.factors))

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
    """Direct sum of one or more sweeps.

    When iterating over a Zip, we iterate the individual sweeps in parallel,
    stopping when the first component sweep stops.
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

    def __iter__(self) -> Iterator[Params]:
        iters = [iter(sweep) for sweep in self.sweeps]
        for values in zip(*iters):
            yield sum(values, ())

    def __repr__(self):
        return 'Zip({})'.format(', '.join(repr(s) for s in self.sweeps))

    def __str__(self):
        if not self.sweeps:
            return 'Zip()'
        return ' + '.join(str(s) for s in self.sweeps)


class _Simple(Sweep):
    """A simple sweep over one parameter with values from an iterator."""

    def __init__(self, key: str) -> None:
        self.key = key

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._tuple())

    @abc.abstractmethod
    def _tuple(self):
        pass

    @property
    def keys(self) -> List[str]:
        return [self.key]

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_func())

    def __iter__(self) -> Iterator[Params]:
        for value in self._iter():
            yield ((self.key, value),)

    @abc.abstractmethod
    def _iter(self) -> Iterator[float]:
        pass


class Points(_Simple):
    """A simple sweep with explicit values."""

    def __init__(self, key: str, points: Sequence[float]) -> None:
        super(Points, self).__init__(key)
        self.points = points

    def _tuple(self):
        return tuple(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def _iter(self) -> Iterator[float]:
        return iter(self.points)

    def __repr__(self):
        return 'Points({!r}, {!r})'.format(self.key, self.points)


class Range(_Simple):
    """A simple sweep over a python-style range."""

    def __init__(self, key, start, stop=None, step=1) -> None:
        super(Range, self).__init__(key)
        if stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step

    def _tuple(self):
        return (self.start, self.stop, self.step)

    def _iter(self) -> Iterator[float]:
        return iter(range(self.start, self.stop, self.step))

    def __repr__(self):
        return 'Range({!r}, start={!r}, stop={!r}, step={!r})'.format(
                self.key, self.start, self.stop, self.step)


class Linspace(_Simple):
    """A simple sweep over linearly-spaced values."""

    def __init__(self, key, start, stop, length) -> None:
        super(Linspace, self).__init__(key)
        self.start = start
        self.stop = stop
        self.length = length

    def _tuple(self):
        return (self.start, self.stop, self.length)

    def __len__(self) -> int:
        return self.length

    def _iter(self) -> Iterator[float]:
        last = self.length - 1
        delta = self.stop - self.start
        for i in range(self.length):
            if i == 0:
                yield self.start
            elif i == last:
                yield self.stop
            else:
                yield self.start + delta * i / last

    def __repr__(self):
        return 'Linspace({!r}, start={!r}, stop={!r}, length={!r})'.format(
                self.key, self.start, self.stop, self.length)
