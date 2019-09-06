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

    `flatten` goes through every parameter in `val` and does the following:
    - If the parameter is a number, don't change it.
    - If the parameter is a symbol, don't change it.
    - If the parameter is an expression, replace it with a symbol.  The new
        symbol will be `sympy.Symbol('<x + 1>')` if the expression was
        `sympy.Symbol('x') + 1`.  In the unlikely case that an expression with a
        different meaning also has the string `'x + 1'`, a number is appended to
        the name to avoid collision: `sympy.Symbol('<x + 1>_1')`.

    This function also creates a dictionary mapping from expressions and symbols
    in `val` to the new symbols in the flattened copy of `val`.  E.g
    `cirq.ExpressionMap({sympy.Symbol('x')+1: sympy.Symbol('<x + 1>')})`.  This
    `ExpressionMap` can be used to transform a sweep over the symbols in `val`
    to a sweep over the flattened symbols e.g. a sweep over `sympy.Symbol('x')`
    to a sweep over `sympy.Symbol('<x + 1>')`.

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
        >>> print(cirq.ListSweep(sweep))
        Sweep:
        {'a': 0.0}
        {'a': 1.0}
        {'a': 2.0}
        {'a': 3.0}

        >>> c_flat, expr_map = cirq.flatten(circuit)
        >>> print(c_flat)
        0: ───X^(<a/4>)───Y^(<1 - a/2>)───
        >>> expr_map
        cirq.ExpressionMap({a/4: <a/4>, 1 - a/2: <1 - a/2>})

        >>> new_sweep = expr_map.transform_sweep(sweep)
        >>> print(new_sweep)
        Sweep:
        {'<a/4>': 0.0, '<1 - a/2>': 1.0}
        {'<a/4>': 0.25, '<1 - a/2>': 0.5}
        {'<a/4>': 0.5, '<1 - a/2>': 0.0}
        {'<a/4>': 0.75, '<1 - a/2>': -0.5}

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

    Args:
        val: The value to copy and substitute parameter expressions with
        flattened symbols.

    Returns:
        The tuple (new value, expression map) where new value and expression map
        are described above.
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
    type.  Also transforms a sweep over the symbols in `val` to a sweep over the
    new symbols.

    `flatten_with_sweep` goes through every parameter in `val` and does the
    following:
    - If the parameter is a number, don't change it.
    - If the parameter is a symbol, don't change it and use the same symbol with
        the same values in the new sweep.
    - If the parameter is an expression, replace it with a symbol and use the
        new symbol with the evaluated value of the expression in the new sweep.
        The new symbol will be `sympy.Symbol('<x + 1>')` if the expression was
        `sympy.Symbol('x') + 1`.  In the unlikely case that an expression with a
        different meaning also has the string `'x + 1'`, a number is appended to
        the name to avoid collision: `sympy.Symbol('<x + 1>_1')`.

    Args:
        val: The value to copy and substitute parameter expressions with
        flattened symbols.
        sweep: A sweep over parameters used by `val`.

    Returns:
        The tuple (new value, new sweep) where new value is `val` with flattened
        expressions and new sweep is the equivalent sweep over it.
    """
    val_flat, expr_map = flatten(val)
    new_sweep = expr_map.transform_sweep(sweep)
    return val_flat, new_sweep


def flatten_with_params(val: Any, params: resolver.ParamResolverOrSimilarType
                       ) -> Tuple[Any, resolver.ParamDictType]:
    """Creates a copy of `val` with any symbols or expressions replaced with
    new symbols.  `val` can be a `Circuit`, `Gate`, `Operation`, or other
    type.  Also transforms a dictionary of symbol values for `val` to an
    equivalent dictionary mapping the new symbols to their evaluated values.

    `flatten_with_params` goes through every parameter in `val` and does the
    following:
    - If the parameter is a number, don't change it.
    - If the parameter is a symbol, don't change it and use the same symbol with
        the same value in the new dictionary of symbol values.
    - If the parameter is an expression, replace it with a symbol and use the
        new symbol with the evaluated value of the expression in the new
        dictionary of symbol values.  The new symbol will be
        `sympy.Symbol('<x + 1>')` if the expression was `sympy.Symbol('x') + 1`.
        In the unlikely case that an expression with a different meaning also
        has the string `'x + 1'`, a number is appended to the name to avoid
        collision: `sympy.Symbol('<x + 1>_1')`.

    Args:
        val: The value to copy and substitute parameter expressions with
        flattened symbols.
        params: A dictionary or `ParamResolver` where the keys are
            `sympy.Symbol`s used by `val` and the values are numbers.

    Returns:
        The tuple (new value, new params) where new value is `val` with
        flattened expressions and new params is a dictionary mapping the
        new symbols like `sympy.Symbol('<x + 1>')` to numbers like
        `params['x'] + 1`.
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
                appended to the name to avoid name collision where `#` is the
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

        - If value is a float, returns it.
        - If value is a str, treat it as a symbol with that name and continue.
        - Otherwise return a symbol unique to the given value.  Return
            `param_dict[value]` if it exists or create a new symbol and add it
            to `param_dict`.

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
        from expression to symbol that is uses on val.

        Args:
            val: The value to copy with substituted parameters.
        """
        return protocols.resolve_parameters(val, self)


class ExpressionMap(dict):
    """A dictionary with sympy expressions and symbols for keys and sympy
    symbols for values.

    This is returned by `cirq.flatten`.  See `ExpressionMap.transform_sweep` and
    `ExpressionMap.transform_params`.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the `ExpressionMap`.

        Takes the same arguments as the builtin `dict`.  Keys must be sympy
        expressions or symbols (instances of `sympy.Basic`).
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
        return sweepable.dict_list_to_sweep([
            dict(param_tuples)
            for param_tuples in _TransformedSweep(sweep, dict(self))
            .param_tuples()
        ])

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
