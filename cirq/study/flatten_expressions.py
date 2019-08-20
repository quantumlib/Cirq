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

from typing import (overload, Any, Callable, Dict, Iterator, List, Optional,
                    Tuple, Union)

import sympy

from cirq import protocols, value
from cirq.study import resolver, sweeps, sweepable


def flatten(val: Any) -> Tuple[Any, 'ExpressionMap']:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.

    Use `ExpressionMap.transform_sweep` or `ExpressionMap.transform_params`
    after `flatten`.  Use `flatten_with_sweep` or `flatten_with_params` instead
    of `flatten` as a shortcut.

    Args:
        val: The value to copy with substituted parameters.

    Example:
        >>> qubit = cirq.LineQubit(0)
        >>> a = sympy.Symbol('a')
        >>> circuit = cirq.Circuit.from_ops(
        ...     cirq.X(qubit) ** (a/4),
        ...     cirq.Y(qubit) ** (1-a/2),
        ... )
        >>> print(circuit)
        0: ───X^(a/4)───Y^(1 - a/2)───

        >>> sweep = cirq.Linspace(a, start=0, stop=3, length=4)
        >>> list(sweep)  #doctest: +NORMALIZE_WHITESPACE
        [cirq.ParamResolver(OrderedDict([('a', 0.0)])),
         cirq.ParamResolver(OrderedDict([('a', 1.0)])),
         cirq.ParamResolver(OrderedDict([('a', 2.0)])),
         cirq.ParamResolver(OrderedDict([('a', 3.0)]))]

        >>> c_flat, expr_map = cirq.flatten(circuit)
        >>> print(c_flat)
        0: ───X^(<a/4>)───Y^(<1 - a/2>)───
        >>> expr_map
        cirq.ExpressionMap({a/4: <a/4>, 1 - a/2: <1 - a/2>})

        >>> new_sweep = expr_map.transform_sweep(sweep)
        >>> list(new_sweep)  #doctest: +NORMALIZE_WHITESPACE
        [cirq.ParamResolver(OrderedDict([('<a/4>', 0.0), ('<1 - a/2>', 1.0)])),
         cirq.ParamResolver(OrderedDict([('<a/4>', 0.25), ('<1 - a/2>', 0.5)])),
         cirq.ParamResolver(OrderedDict([('<a/4>', 0.5), ('<1 - a/2>', 0.0)])),
         cirq.ParamResolver(OrderedDict([('<a/4>', 0.75), ('<1 - a/2>', -0.5)])\
        )]

        >>> for params in sweep:  # Original
        ...     print(circuit, '=>', end=' ')
        ...     print(cirq.resolve_parameters(circuit, params))
        0: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0───Y───
        0: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.25───Y^0.5───
        0: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.5───Y^0───
        0: ───X^(a/4)───Y^(1 - a/2)─── => 0: ───X^0.75───Y^-0.5───
        >>> for params in new_sweep:  # Flattened
        ...     print(c_flat, '=>', end=' ')
        ...     print(cirq.resolve_parameters(c_flat, params))
        0: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0───Y───
        0: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.25───Y^0.5───
        0: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.5───Y^0───
        0: ───X^(<a/4>)───Y^(<1 - a/2>)─── => 0: ───X^0.75───Y^-0.5───

    Returns:
        A tuple containing the value with flattened parameters and the
        expression map mapping expressions in `val` to symbols.
    """
    flattener = _ParamFlattener()
    val_flat = flattener.flatten(val)
    expr_map = ExpressionMap(flattener.param_dict)
    return val_flat, expr_map


def flatten_with_sweep(val: Any,
                       sweep: Union[sweeps.Sweep, List[resolver.ParamResolver]]
                      ) -> Tuple[Any, sweeps.Sweep]:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.  Also creates a transformed `Sweep` that resolves the new value.

    Args:
        val: The value to copy with substituted parameters.
        sweep: A sweep over parameters used by val.

    Returns:
        A tuple containing the value with flattened parameters and the new
        sweep.
    """
    val_flat, expr_map = flatten(val)
    new_sweep = expr_map.transform_sweep(sweep)
    return val_flat, new_sweep


def flatten_with_params(val: Any, params: resolver.ParamResolverOrSimilarType
                       ) -> Tuple[Any, resolver.ParamDictType]:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.  Also creates transformed parameters that resolve the new value.

    Args:
        val: The value to copy with substituted parameters.
        params: Parameters that map symbols used by val.

    Returns:
        A tuple containing the value with flattened parameters and the
        transformed parameters.
    """
    val_flat, expr_map = flatten(val)
    new_params = expr_map.transform_params(params)
    return val_flat, new_params


class _ParamFlattener(resolver.ParamResolver):
    """A `ParamResolver` that resolves sympy expressions to unique symbols.

    This is a mutable object that stores new expression to symbol mappings
    when it is used to resolve parameters with `cirq.resolve_parameters` or
    `_ParamFlattener.flatten_circuit`.  It is useful for replacing sympy
    expressions from circuits with single symbols and transforming parameter
    sweeps to match.
    """

    def __new__(cls, *args, **kwargs):
        """Disables the behavior of `ParamResolver.__new__`."""
        return super().__new__(cls)

    def __init__(
            self,
            param_dict: Optional[resolver.ParamResolverOrSimilarType] = None,
            *,  # Force keyword args
            get_param_name: Callable[[
                sympy.Basic,
            ], str] = None):
        """Initializes a new _ParamFlattener.

        Args:
            param_dict: A default initial mapping from some parameter names,
                symbols, or expressions to other symbols or values.  Only sympy
                expressions and symbols not specified in `param_dict` will be
                flattened.
            get_param_name: A callback function that returns a new parameter
                name for a given sympy expression or symbol.  If this function
                returns the same value for two different expressions, `'_#'` is
                appended to the name to avoid name collision wher `#` is the
                number of previous collisions.  By default, returns the
                expression string surrounded by angle brackets e.g. `'<x+1>'`.
        """
        if hasattr(self, '_taken_symbols'):
            # Has already been initialized
            return
        if isinstance(param_dict, resolver.ParamResolver):
            params = param_dict.param_dict
        else:
            params = param_dict if param_dict else {}
        symbol_params = {
            _ensure_not_str(param): _ensure_not_str(val)
            for param, val in params.items()
        }
        super().__init__(symbol_params)
        if get_param_name is None:
            get_param_name = self.default_get_param_name
        self.get_param_name = get_param_name
        self._taken_symbols = set(self.param_dict.values())

    @staticmethod
    def default_get_param_name(val: sympy.Basic) -> str:
        if isinstance(val, sympy.Symbol):
            return val.name
        return '<{!s}>'.format(val)

    def _next_symbol(self, val: sympy.Basic) -> sympy.Symbol:
        name = self.get_param_name(val)
        symbol = sympy.Symbol(name)
        # Ensure the symbol hasn't already been used
        collision = 0
        while symbol in self._taken_symbols:
            collision += 1
            symbol = sympy.Symbol('{}_{}'.format(name, collision))
        return symbol

    def value_of(self, value: Union[sympy.Basic, float, str]
                ) -> Union[sympy.Basic, float]:
        """Resolves a symbol or expression to a new symbol unique to that value.

        If value is a float, returns it.
        If value is a str, treat it as a symbol with that name and continue.
        If this `_ParamFlattener` was initialized with a `param_dict` and
        `value` is a key, returns `param_dict[value]`.
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
        symbol = self._next_symbol(value)
        self.param_dict[value] = symbol
        self._taken_symbols.add(symbol)
        return symbol

    # Default object truth, equality, and hash
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__

    def __bool__(self):
        return True

    def __repr__(self):
        if self.get_param_name == self.default_get_param_name:
            return '_ParamFlattener({!r})'.format(self.param_dict)
        else:
            return ('_ParamFlattener({!r}, get_param_name={!r})'.format(
                self.param_dict, self.get_param_name))

    def flatten(self, val: Any) -> Any:
        """Returns a copy of `val` with any symbols or expressions replaced with
        new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
        type.

        This method mutates the `_ParamFlattener` by storing any new mappings
        from expression to symbol that is uses on val.  Use `transform_sweep` or
        `transform_params` after `flatten`.

        Args:
            val: The value to copy with substituted parameters.
        """
        return protocols.resolve_parameters(val, self)


class ExpressionMap(dict):
    """A dictionary where the keys are sympy expressions and symbols and the
    values are sympy symbols.

    This is returned by `cirq.flatten`.  See `ExpressionMap.transform_sweep` and
    `ExpressionMap.transform_params`.
    """

    def __init__(self, *args, **kwargs):
        """Initialized the `ExpressionMap`.  The arguments are the same as the
        builtin `dict`.
        """
        super().__init__(*args, **kwargs)

    def transform_sweep(self,
                        sweep: Union[sweeps.Sweep, List[resolver.ParamResolver]]
                       ) -> sweeps.Sweep:
        """Returns a sweep to use with a circuit flattened earlier with
        `cirq.flatten`.

        If `sweep` sweeps symbol `a` over (1.0, 2.0, 3.0) and this
        `ExpressionMap` maps `a/2+1` to the symbol `'<a/2 + 1>'` then this
        method returns a sweep that sweeps symbol `'<a/2 + 1>'` over
        (1.5, 2, 2.5).

        See `cirq.flatten` for an example.

        Args:
            sweep: The sweep to transform.
        """
        sweep = sweepable.to_sweep(sweep)
        return _TransformedSweep(sweep, dict(self))

    def transform_params(self, params: resolver.ParamResolverOrSimilarType
                        ) -> resolver.ParamDictType:
        """Returns a `ParamResolver` to use with a circuit flattened earlier
        with `cirq.flatten`.

        If `params` maps symbol `a` to 3.0 and this `ExpressionMap` maps
        `a/2+1` to `'<a/2 + 1>'` then this method returns a resolver that maps
        symbol `'<a/2 + 1>'` to 2.5.

        See `cirq.flatten` for an example.

        Args:
            params: The params to transform.
        """
        param_dict = {
            sym: protocols.resolve_parameters(formula, params)
            for formula, sym in self.items()
            if isinstance(sym, sympy.Basic)
        }
        return param_dict

    def __repr__(self):
        return 'cirq.ExpressionMap({})'.format(super().__repr__())


@value.value_equality(unhashable=True)
class _TransformedSweep(sweeps.Sweep):
    """A sweep created by `ExpressionMap.transform_sweep`."""

    def __init__(self, sweep: sweeps.Sweep,
                 expression_map: Dict[Union[sympy.Basic, float],
                                      Union[sympy.Basic, float]]):
        self.sweep = sweep
        self.expression_map = expression_map

    def _value_equality_values_(self):
        return self.sweep, self.expression_map

    # The value_equality decorator implements the __eq__ method after the ABC
    # meta class had recorded that __eq__ is an abstract method.
    def __eq__(self, other):
        raise NotImplementedError  # coverage: ignore

    @property
    def keys(self) -> List[str]:
        return [
            str(sym)
            for sym in self.expression_map.values()
            if isinstance(sym, (sympy.Symbol, str))
        ]

    def __len__(self) -> int:
        return len(self.sweep)

    def param_tuples(self) -> Iterator[sweeps.Params]:
        for r in self.sweep:
            yield tuple((str(sym), protocols.resolve_parameters(formula, r))
                        for formula, sym in self.expression_map.items()
                        if isinstance(sym, (sympy.Symbol, str)))


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
