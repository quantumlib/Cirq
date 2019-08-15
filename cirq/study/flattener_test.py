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

import pytest
import sympy
import cirq


def test_flatten_circuit_sweep():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit.from_ops(
        cirq.X(qubit)**(a / 4),
        cirq.X(qubit)**(1 - a / 4),
    )

    flattener = cirq.ParamFlattener()
    c_flat = flattener.flatten(circuit)

    c_expected = cirq.Circuit.from_ops(
        cirq.X(qubit)**sympy.Symbol('x0'),
        cirq.X(qubit)**sympy.Symbol('x1'),
    )
    assert c_flat == c_expected
    assert flattener.param_dict == {
        a / 4: sympy.Symbol('x0'),
        1 - a / 4: sympy.Symbol('x1')
    }


def test_transform_resolver():
    a = sympy.Symbol('a')
    flattener = cirq.ParamFlattener({
        a / 4: 'x0',  # Either a str or a Symbol
        1 - a / 4: sympy.Symbol('x1')
    })
    resolver = cirq.ParamResolver({'a': 3})

    new_resolver = flattener.transform_resolver(resolver)
    assert isinstance(new_resolver, cirq.ParamResolver)

    expected_resolver = cirq.ParamResolver({
        sympy.Symbol('x0'): 3 / 4,
        sympy.Symbol('x1'): 1 - 3 / 4
    })
    assert new_resolver == expected_resolver


def test_transform_sweep():
    a = sympy.Symbol('a')
    flattener = cirq.ParamFlattener({
        a / 4: sympy.Symbol('x0'),
        1 - a / 4: sympy.Symbol('x1')
    })
    sweep = cirq.Linspace(a, start=0, stop=3, length=4)

    new_sweep = flattener.transform_sweep(sweep)
    assert isinstance(new_sweep, cirq.Sweep)
    resolvers = list(new_sweep)

    expected_resolvers = [
        cirq.ParamResolver({
            'x0': 0.0,
            'x1': 1.0
        }),
        cirq.ParamResolver({
            'x0': 0.25,
            'x1': 0.75
        }),
        cirq.ParamResolver({
            'x0': 0.5,
            'x1': 0.5
        }),
        cirq.ParamResolver({
            'x0': 0.75,
            'x1': 0.25
        })
    ]
    assert resolvers == expected_resolvers


def test_flattener_new():
    flattener = cirq.ParamFlattener({'a': 'b'})
    flattener2 = cirq.ParamFlattener(flattener)
    assert isinstance(flattener2, cirq.ParamFlattener)
    assert flattener2.param_dict == flattener.param_dict


def test_resolver_new():
    flattener = cirq.ParamFlattener({'a': 'b'})
    flattener2 = cirq.ParamResolver(flattener)
    assert flattener2 is flattener


def test_transformed_sweep():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    flattener = cirq.ParamFlattener({
        a/4: 'x0',
        1-a/4: 'x1'})
    transformed = flattener.transform_sweep(sweep)
    assert len(transformed) == 4
    assert transformed.keys == ['x0', 'x1']
    params = list(transformed.param_tuples())
    assert len(params) == 4
    assert params[1] == (('x0', 1/4), ('x1', 1-1/4))


def test_transformed_sweep_equality():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    flattener = cirq.ParamFlattener({
        a/4: 'x0',
        1-a/4: 'x1'})

    sweep2 = cirq.Linspace(a, start=0, stop=3, length=4)
    flattener2 = cirq.ParamFlattener({
        a/4: 'x0',
        1-a/4: 'x1'})

    sweep3 = cirq.Linspace(a, start=0, stop=3, length=20)
    flattener3 = cirq.ParamFlattener({
        a/20: 'x0',
        1-a/20: 'x1'})

    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: flattener.transform_sweep(sweep),
                           lambda: flattener.transform_sweep(sweep2),
                           lambda: flattener2.transform_sweep(sweep2))
    et.add_equality_group(flattener.transform_sweep(sweep3))
    et.add_equality_group(flattener3.transform_sweep(sweep))
    et.add_equality_group(flattener3.transform_sweep(sweep3))
