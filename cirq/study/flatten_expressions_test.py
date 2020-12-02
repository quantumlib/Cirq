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

import sympy
import cirq
from cirq.study import flatten_expressions


# None of the following tests use expressions of the form
# <constant term> - <other term> because the string of expressions containing
# exactly two terms, one constant term and one non-constant term with a negative
# factor, is not consistent between sympy versions <1.4 and >=1.4.


def test_expr_map_names():
    flattener = flatten_expressions._ParamFlattener({'collision': '<x + 2>'})
    expressions = [sympy.Symbol('x') + i for i in range(3)]
    syms = flattener.flatten(expressions)
    assert syms == [sympy.Symbol(name) for name in ('x', '<x + 1>', '<x + 2>_1')]


def test_flattener_value_of():
    flattener = flatten_expressions._ParamFlattener({'c': 5, 'x1': 'x1'})
    assert flattener.value_of(9) == 9
    assert flattener.value_of('c') == 5
    assert flattener.value_of(sympy.Symbol('c')) == 5
    # Twice
    assert flattener.value_of(sympy.Symbol('c') / 2 + 1) == sympy.Symbol('<c/2 + 1>')
    assert flattener.value_of(sympy.Symbol('c') / 2 + 1) == sympy.Symbol('<c/2 + 1>')
    # Collisions between the string representation of different expressions
    # This tests the unusual case where str(expr1) == str(expr2) doesn't imply
    # expr1 == expr2.  In this case it would be incorrect to flatten to the same
    # symbol because the two expression will evaluate to different values.
    # Also tests that '_#' is appended when avoiding collisions.
    assert flattener.value_of(sympy.Symbol('c') / sympy.Symbol('2 + 1')) == sympy.Symbol(
        '<c/2 + 1>_1'
    )
    assert flattener.value_of(sympy.Symbol('c/2') + 1) == sympy.Symbol('<c/2 + 1>_2')

    assert cirq.flatten([sympy.Symbol('c') / 2 + 1, sympy.Symbol('c/2') + 1])[0] == [
        sympy.Symbol('<c/2 + 1>'),
        sympy.Symbol('<c/2 + 1>_1'),
    ]


def test_flattener_repr():
    assert repr(flatten_expressions._ParamFlattener({'a': 1})) == ("_ParamFlattener({a: 1})")
    assert repr(
        flatten_expressions._ParamFlattener({'a': 1}, get_param_name=lambda expr: 'x')
    ).startswith("_ParamFlattener({a: 1}, get_param_name=<function ")


def test_expression_map_repr():
    cirq.testing.assert_equivalent_repr(cirq.ExpressionMap({'a': 'b'}))


def test_flatten_circuit():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(
        cirq.X(qubit) ** a,
        cirq.X(qubit) ** (1 + a / 2),
    )

    c_flat, expr_map = cirq.flatten(circuit)

    c_expected = cirq.Circuit(
        cirq.X(qubit) ** a,
        cirq.X(qubit) ** sympy.Symbol('<a/2 + 1>'),
    )
    assert c_flat == c_expected
    assert isinstance(expr_map, cirq.ExpressionMap)
    assert expr_map == {
        a: a,
        1 + a / 2: sympy.Symbol('<a/2 + 1>'),
    }


def test_transform_params():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(
        cirq.X(qubit) ** (a / 4),
        cirq.X(qubit) ** (1 + a / 2),
    )
    params = {'a': 3}

    _, new_params = cirq.flatten_with_params(circuit, params)

    expected_params = {sympy.Symbol('<a/4>'): 3 / 4, sympy.Symbol('<a/2 + 1>'): 1 + 3 / 2}
    assert new_params == expected_params


def test_transform_sweep():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(
        cirq.X(qubit) ** (a / 4),
        cirq.X(qubit) ** (1 + a / 2),
    )
    sweep = cirq.Linspace(a, start=0, stop=3, length=4)

    _, new_sweep = cirq.flatten_with_sweep(circuit, sweep)
    assert isinstance(new_sweep, cirq.Sweep)
    resolvers = list(new_sweep)

    expected_resolvers = [
        cirq.ParamResolver(
            {
                '<a/4>': 0.0,
                '<a/2 + 1>': 1.0,
            }
        ),
        cirq.ParamResolver(
            {
                '<a/4>': 0.25,
                '<a/2 + 1>': 1.5,
            }
        ),
        cirq.ParamResolver(
            {
                '<a/4>': 0.5,
                '<a/2 + 1>': 2,
            }
        ),
        cirq.ParamResolver(
            {
                '<a/4>': 0.75,
                '<a/2 + 1>': 2.5,
            }
        ),
    ]
    assert resolvers == expected_resolvers


def test_flattener_new():
    flattener = flatten_expressions._ParamFlattener({'a': 'b'})
    flattener2 = flatten_expressions._ParamFlattener(flattener)
    assert isinstance(flattener2, flatten_expressions._ParamFlattener)
    assert flattener2.param_dict == flattener.param_dict


def test_resolver_new():
    flattener = flatten_expressions._ParamFlattener({'a': 'b'})
    flattener2 = cirq.ParamResolver(flattener)
    assert flattener2 is flattener


def test_transformed_sweep():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    expr_map = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 2: 'x1'})
    transformed = expr_map.transform_sweep(sweep)
    assert len(transformed) == 4
    assert transformed.keys == ['x0', 'x1']
    params = list(transformed.param_tuples())
    assert len(params) == 4
    assert params[1] == (('x0', 1 / 4), ('x1', 1 - 1 / 2))


def test_transformed_sweep_equality():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    expr_map = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 4: 'x1'})

    sweep2 = cirq.Linspace(a, start=0, stop=3, length=4)
    expr_map2 = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 4: 'x1'})

    sweep3 = cirq.Linspace(a, start=0, stop=3, length=20)
    expr_map3 = cirq.ExpressionMap({a / 20: 'x0', 1 - a / 20: 'x1'})

    et = cirq.testing.EqualsTester()
    et.make_equality_group(
        lambda: expr_map.transform_sweep(sweep),
        lambda: expr_map.transform_sweep(sweep2),
        lambda: expr_map2.transform_sweep(sweep2),
    )
    et.add_equality_group(expr_map.transform_sweep(sweep3))
    et.add_equality_group(expr_map3.transform_sweep(sweep))
    et.add_equality_group(expr_map3.transform_sweep(sweep3))
