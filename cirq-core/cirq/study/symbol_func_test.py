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

import sympy

import cirq


def test_compile_expr():
    assert cirq.SymbolFunc.compile_expr(123) == 123
    assert cirq.SymbolFunc.compile_expr(sympy.Symbol('abc')) == sympy.Symbol('abc')

    expr = sympy.Symbol('abc') + 1
    compiled = cirq.SymbolFunc.compile_expr(expr)
    assert isinstance(compiled, cirq.SymbolFunc)
    assert compiled.expr == expr


def test_symbol_func_repr():
    assert repr(cirq.SymbolFunc(sympy.Symbol('a'))) == "cirq.SymbolFunc(sympy.Symbol('a'))"


def test_resolve_parameters():
    func = cirq.SymbolFunc(sympy.Symbol('a') + sympy.Symbol('b'))
    assert cirq.is_parameterized(func)
    assert cirq.parameter_names(func) == {'a', 'b'}
    assert cirq.resolve_parameters(func, {'a': 1, 'b': 10}) == 11

    expr2 = cirq.resolve_parameters(func, {'a': 1})
    assert isinstance(expr2, sympy.Basic)
    assert cirq.is_parameterized(expr2)
    assert cirq.parameter_names(expr2) == {'b'}
    assert cirq.resolve_parameters(expr2, {'b': 10}) == 11
