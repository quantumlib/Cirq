# Copyright 2022 The Cirq Developers
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

from typing import AbstractSet, TYPE_CHECKING, TypeVar, Union

import sympy

import cirq._compat as _compat
import cirq.protocols as protocols

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
        return f'cirq.SymbolFunc({_compat.proper_repr(self.expr)})'

    def _is_parameterized_(self) -> bool:
        return True

    def _parameter_names_(self) -> AbstractSet[str]:
        return self.param_set

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> float:
        args = [resolver.value_of(param, recursive=recursive) for param in self.params]
        return self.func(*args)
