# Copyright 2019 The Cirq Developers
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
"""Resolves symbolic expressions to unique symbols."""

from typing import overload, Any, Dict, Iterable, Iterator, List, Union

import itertools
import sympy

from cirq.study.resolver import ParamResolver, ParamResolverOrSimilarType
from cirq.study.sweeps import Sweep, Params, _params_without_symbols
from cirq.study.sweepable import to_sweep
from cirq.protocols.resolve_parameters import resolve_parameters


class ParamFlattener(ParamResolver):
    """A `ParamResolver` that resolves sympy expressions to unique symbols.

    This is a mutable object that stores new expression to symbol mappings
    when it is used to resolve parameters with `cirq.resolve_parameters` or
    `ParamFlattener.flatten_circuit`.  It is useful for replacing sympy
    expressions from circuits with single symbols and transforming parameter
    sweeps to match.

    Example:
        >>> qubit = cirq.LineQubit(0)
        >>> a = sympy.Symbol('a')
        >>> circuit = cirq.Circuit.from_ops(
        ...     cirq.X(qubit) ** (a/4),
        ...     cirq.X(qubit) ** (1-a/4),
        ... )
        >>> print(circuit)
        0: ───X^(a/4)───X^(1 - a/4)───

        >>> sweep = cirq.Linspace(a, start=0, stop=3, length=4)
        >>> list(sweep)  #doctest: +NORMALIZE_WHITESPACE
        [cirq.ParamResolver(OrderedDict([('a', 0.0)])),
         cirq.ParamResolver(OrderedDict([('a', 1.0)])),
         cirq.ParamResolver(OrderedDict([('a', 2.0)])),
         cirq.ParamResolver(OrderedDict([('a', 3.0)]))]

        >>> flattener = cirq.ParamFlattener()
        >>> c_flat = flattener.flatten(circuit)
        >>> print(c_flat)
        0: ───X^x0───X^x1───

        >>> new_sweep = flattener.transform_sweep(sweep)
        >>> list(new_sweep)  #doctest: +NORMALIZE_WHITESPACE
        [cirq.ParamResolver({x0: 0.0, x1: 1.0}),
         cirq.ParamResolver({x0: 0.25, x1: 0.75}),
         cirq.ParamResolver({x0: 0.5, x1: 0.5}),
         cirq.ParamResolver({x0: 0.75, x1: 0.25})]

        >>> print(cirq.resolve_parameters(c_flat, list(new_sweep)[1]))
        0: ───X^0.25───X^0.75───
        >>> print(cirq.resolve_parameters(circuit, list(sweep)[1])) # Equivalent
        0: ───X^0.25───X^0.75───
    """

    def __new__(cls, *args, **kwargs):
        """Disables the behavior of `ParamResolver.__new__`."""
        return super().__new__(cls)

    def __init__(self,
                 param_dict: ParamResolverOrSimilarType = None,
                 new_param_names: Iterable[str] = ()):
        """Initializes a new ParamFlattener.

        Args:
            param_dict: A default initial mapping from some parameter names,
                symbols, or expressions to other symbols or values.  Only sympy
                expressions and symbols not specified in `param_dict` will be
                flattened.
            new_param_names: An iterator of parameter names to replace each
                expression with.  It may be infinite but there must be no
                repetitions.  By default the parameter names are x0, x1, x2, ...
        """
        if isinstance(param_dict, ParamResolver):
            param_dict = {
                _ensure_not_str(param): _ensure_not_str(val)
                for param, val in param_dict.param_dict.items()
            }
        super().__init__(param_dict)
        if new_param_names is None:
            new_param_names = ParamFlattener.default_param_names()
        self._param_name_gen = iter(new_param_names)
        self.new_param_names = new_param_names
        self._taken_symbols = set(self.param_dict.values())

    @staticmethod
    def default_param_names(prefix: str = 'x', start: int = 0) -> Iterator[str]:
        """Generates an infinite sequence of parameter names with the format
        '<prefix><index>' with no repetitions.  E.g. x0, x1, x2, ...

        Args:
            prefix: The prefix string of the parameter name.
            start: The index of the first parameter name generated.

        Returns:
            An infinite iterator of parameter names.
        """
        for i in itertools.count(start):
            yield '{}{}'.format(prefix, i)

    def _next_name(self) -> str:
        try:
            return next(self._param_name_gen)
        except StopIteration:
            self._param_name_gen = self.default_param_names()
            return next(self._param_name_gen)

    def value_of(self, value: Union[sympy.Basic, float, str]
                ) -> Union[sympy.Basic, float]:
        """Resolves a symbol or expression to a new symbol unique to that value.

        If value is a float, returns it.
        If this `ParamFlattener` was initialized with a `param_dict` and `value`
        is a key, returns `param_dict[value]`.
        Otherwise return a symbol unique to the given value.

        Args:
            value: The sympy.Symbol, sympy expression, name, or float to resolve
                to a unique symbol or float.

        Returns:
            The unique symbol or value of the parameter as resolved by this
            resolver.
        """
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            value = sympy.Symbol(value)
        out = self.param_dict.get(value, None)
        if out is not None:
            return out
        # Create a new symbol
        symbol = sympy.Symbol(self._next_name())
        # Ensure the symbol hasn't already been used
        while symbol in self._taken_symbols:
            symbol = sympy.Symbol(self._next_name())
        self.param_dict[value] = symbol
        self._taken_symbols.add(symbol)
        return symbol

    # Default object equality and hash
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__

    def __repr__(self):
        return ('cirq.ParamFlattener({!r}, new_param_names={!r})'.format(
            self.param_dict, self.new_param_names))

    def flatten(self, val: Any) -> Any:
        """Returns a copy of `val` with any symbols or expressions replaced with
        new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
        type.

        This method mutates the `ParamFlattener` by storing any new mappings
        from expression to symbol that is uses on val.  Use `transform_sweep` or
        `transform_resolver` after `flatten`.

        Args:
            val: The value to copy with substituted parameters.
        """
        return resolve_parameters(val, self)

    def transform_sweep(self,
                        sweep: Union[Sweep, List[ParamResolver]]) -> Sweep:
        """Returns a sweep to use with a circuit flattened earlier with
        `flatten`.

        If `sweep` sweeps symbol `a` over (1.0, 2.0, 3.0) and this
        `ParamFlattener` maps `a/2+1` to `x0` then this method returns a sweep
        that sweeps symbol `x0` over (1.5, 2, 2.5).

        See the class doc for an example.

        Args:
            sweep: The sweep to transform.
        """
        sweep = to_sweep(sweep)
        return _TransformedSweep(sweep, dict(self.param_dict))

    def transform_resolver(self, resolver: ParamResolverOrSimilarType
                          ) -> ParamResolver:
        """Returns a `ParamResolver` to use with a circuit flattened earlier
        with `flatten`.

        If `resolver` maps symbol `a` to 3.0 and this `ParamFlattener` maps
        `a/2+1` to `x0` then this method returns a resolver that maps symbol
        `x0` to 2.5.

        See the class doc for example code.

        Args:
            resolver: The resolver to transform.
        """
        return _transform_resolver(resolver, self.param_dict)


class _TransformedSweep(Sweep):
    """A sweep created by `ParamFlattener.transform_sweep`."""

    def __init__(self, sweep: Sweep,
                 expression_map: Dict[Union[sympy.Basic, float],
                                      Union[sympy.Basic, float]]):
        self.sweep = sweep
        self.expression_map = expression_map

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.sweep == other.sweep and
                self.expression_map == other.expression_map)

    def __ne__(self, other):
        return not self == other

    @property
    def keys(self) -> List[str]:
        return [
            sym for sym in self.expression_map.values()
            if isinstance(sym, sympy.Basic)
        ]

    def __len__(self) -> int:
        return len(self.sweep)

    def __iter__(self) -> Iterator[ParamResolver]:
        for r in self.sweep:
            yield _transform_resolver(r, self.expression_map)

    def param_tuples(self) -> Iterator[Params]:
        for r in self:
            yield _params_without_symbols(r)


def _transform_resolver(
        resolver: ParamResolverOrSimilarType,
        expression_map: Dict[Union[sympy.Basic, float], Union[sympy.
                                                              Basic, float]]
) -> ParamResolver:
    param_dict = {
        sym: resolve_parameters(formula, resolver)
        for formula, sym in expression_map.items()
        if isinstance(sym, sympy.Basic)
    }
    return ParamResolver(param_dict)


@overload
def _ensure_not_str(param: Union[sympy.Basic, str]) -> sympy.Basic:
    pass


@overload
def _ensure_not_str(param: float) -> float:
    pass


def _ensure_not_str(param):
    if isinstance(param, str):
        return sympy.Symbol(param)
    return param
