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
from typing import (Any, cast, Dict, Iterable, Iterator, List, overload,
                    Sequence, Tuple, Union)

import abc
import collections
import itertools
import sympy

from cirq._doc import document
from cirq.study import resolver


Params = Iterable[Tuple[str, float]]


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
    This can then be used with a circuit that has an 'angle' sympy.Symbol to
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
            raise TypeError(f'cannot multiply sweep and {type(other)}')
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
            raise TypeError(f'cannot add sweep and {type(other)}')
        return Zip(*sweeps)

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self == other

    @property
    @abc.abstractmethod
    def keys(self) -> List[str]:
        """The keys for the all of the sympy.Symbols that are resolved."""

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[resolver.ParamResolver]:
        for params in self.param_tuples():
            yield resolver.ParamResolver(collections.OrderedDict(params))

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, val: int) -> resolver.ParamResolver:
        pass

    @overload
    def __getitem__(self, val: slice) -> 'Sweep':
        pass

    def __getitem__(self, val):
        n = len(self)
        if isinstance(val, int):
            if val < -n or val >= n:
                raise IndexError(f'sweep index out of range: {val}')
            if val < 0:
                val += n
            return next(itertools.islice(self, val, val + 1))
        if not isinstance(val, slice):
            raise TypeError(
                f'Sweep indices must be either int or slices, not {type(val)}')

        inds_map: Dict[int, int] = {
            sweep_i: slice_i for slice_i, sweep_i in enumerate(range(n)[val])
        }
        results = [resolver.ParamResolver()] * len(inds_map)
        for i, item in enumerate(self):
            if i in inds_map:
                results[inds_map[i]] = item

        return ListSweep(results)

    # pylint: enable=function-redefined

    @abc.abstractmethod
    def param_tuples(self) -> Iterator[Params]:
        """An iterator over (key, value) pairs assigning Symbol key to value."""

    def __str__(self) -> str:
        length = len(self)
        max_show = 10
        # Show a maximum of max_show entries with an ellipsis in the middle
        if length > max_show:
            beginning_len = max_show - max_show // 2
        else:
            beginning_len = max_show
        end_len = max_show - beginning_len
        lines = ['Sweep:']
        lines.extend(
            str(dict(r.param_dict))
            for r in itertools.islice(self, beginning_len))
        if end_len > 0:
            lines.append('...')
            lines.extend(
                str(dict(r.param_dict))
                for r in itertools.islice(self, length - end_len, length))
        return '\n'.join(lines)


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

    def __repr__(self) -> str:
        return 'cirq.UnitSweep'


UnitSweep = _Unit()
document(UnitSweep, """The singleton sweep with no parameters.""")


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

    def __repr__(self) -> str:
        factors_repr = ', '.join(repr(f) for f in self.factors)
        return f'cirq.Product({factors_repr})'

    def __str__(self) -> str:
        if not self.factors:
            return 'Product()'
        factor_strs = []
        for factor in self.factors:
            factor_str = repr(factor)
            if isinstance(factor, Zip):
                factor_str = '(' + str(factor) + ')'
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

    def __hash__(self) -> int:
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

    def __repr__(self) -> str:
        sweeps_repr = ', '.join(repr(s) for s in self.sweeps)
        return f'cirq.Zip({sweeps_repr})'

    def __str__(self) -> str:
        if not self.sweeps:
            return 'Zip()'
        return ' + '.join(
            str(s) if isinstance(s, Product) else repr(s) for s in self.sweeps)


class SingleSweep(Sweep):
    """A simple sweep over one parameter with values from an iterator."""

    def __init__(self, key: Union[str, sympy.Symbol]) -> None:
        if isinstance(key, sympy.Symbol):
            key = str(key)
        self.key = key

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __hash__(self) -> int:
        return hash((self.__class__, self._tuple()))

    @abc.abstractmethod
    def _tuple(self) -> Tuple[Any, ...]:
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

    def __init__(
        self, key: Union[str, sympy.Symbol],
        points: Sequence[float]) -> None:
        super(Points, self).__init__(key)
        self.points = points

    def _tuple(self) -> Tuple[Union[str, sympy.Symbol], Sequence[float]]:
        return self.key, tuple(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def _values(self) -> Iterator[float]:
        return iter(self.points)

    def __repr__(self) -> str:
        return f'cirq.Points({self.key!r}, {self.points!r})'


class Linspace(SingleSweep):
    """A simple sweep over linearly-spaced values."""

    def __init__(
        self, key: Union[str, sympy.Symbol],
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

    def _tuple(self) -> Tuple[Union[str, sympy.Symbol], float, float, int]:
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

    def __repr__(self) -> str:
        return (f'cirq.Linspace({self.key!r}, start={self.start!r}, '
                f'stop={self.stop!r}, length={self.length!r})')


class ListSweep(Sweep):
    """A wrapper around a list of `ParamResolver`s."""

    def __init__(self,
                 resolver_list: Iterable[resolver.ParamResolverOrSimilarType]):
        """Creates a `Sweep` over a list of `ParamResolver`s.

        Args:
            resolver_list: The list of parameter resolvers to use in the sweep.
                All resolvers must resolve the same set of parameters.
        """
        self.resolver_list: List[resolver.ParamResolver] = []
        for r in resolver_list:
            if not isinstance(r, (dict, resolver.ParamResolver)):
                raise TypeError(f'Not a ParamResolver or dict: <{r!r}>')
            self.resolver_list.append(resolver.ParamResolver(r))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.resolver_list == other.resolver_list

    def __ne__(self, other):
        return not self == other

    @property
    def keys(self) -> List[str]:
        if not self.resolver_list:
            return []
        return list(map(str, self.resolver_list[0].param_dict))

    def __len__(self) -> int:
        return len(self.resolver_list)

    def param_tuples(self) -> Iterator[Params]:
        for r in self.resolver_list:
            yield tuple(_params_without_symbols(r))

    def __repr__(self) -> str:
        return f'cirq.ListSweep({self.resolver_list!r})'


def _params_without_symbols(resolver: resolver.ParamResolver) -> Params:
    for sym, val in resolver.param_dict.items():
        if isinstance(sym, sympy.Symbol):
            sym = sym.name
        yield cast(str, sym), cast(float, val)
