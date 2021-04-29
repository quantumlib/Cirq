from typing import AbstractSet, TYPE_CHECKING, TypeVar, Union

import sympy

from cirq import protocols

if TYPE_CHECKING:
    import cirq


T = TypeVar('T')

class SymbolFunc:
    """A "lambdified" symbolic expression that is faster for parameter resolution."""

    @classmethod
    def compile_expr(cls, expr: Union[T, sympy.Basic]) -> Union[T, sympy.Symbol, 'SymbolFunc']:
        if isinstance(expr, sympy.Symbol):
            return expr
        if isinstance(expr, sympy.Basic):
            return cls(expr)
        return expr

    def __init__(self, expr: sympy.Basic) -> None:
        self.expr = expr
        self.param_set = protocols.parameter_names(expr)
        self.params = sorted(self.param_set)
        self.func = sympy.lambdify(self.params, expr)

    def __repr__(self) -> str:
        return f'SymbolFunc({self.expr!r})'

    def _is_parameterized_(self) -> bool:
        return True

    def _parameter_names_(self) -> AbstractSet[str]:
        return self.param_set

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> float:
        args = [resolver.value_of(param, recursive=recursive) for param in self.params]
        return self.func(*args)
