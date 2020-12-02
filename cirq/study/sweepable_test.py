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
"""Tests for sweepable.py."""

import itertools

import pytest
import sympy

import cirq


def test_to_resolvers_none():
    assert list(cirq.to_resolvers(None)) == [cirq.ParamResolver({})]


def test_to_resolvers_single():
    resolver = cirq.ParamResolver({})
    assert list(cirq.to_resolvers(resolver)) == [resolver]
    assert list(cirq.to_resolvers({})) == [resolver]


def test_to_resolvers_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert list(cirq.to_resolvers(sweep)) == list(sweep)


def test_to_resolvers_iterable():
    resolvers = [cirq.ParamResolver({'a': 2}), cirq.ParamResolver({'a': 1})]
    assert list(cirq.to_resolvers(resolvers)) == resolvers
    assert list(cirq.to_resolvers([{'a': 2}, {'a': 1}])) == resolvers


def test_to_resolvers_iterable_sweeps():
    sweeps = [cirq.Linspace('a', 0, 1, 10), cirq.Linspace('b', 0, 1, 10)]
    assert list(cirq.to_resolvers(sweeps)) == list(itertools.chain(*sweeps))


def test_to_resolvers_bad():
    with pytest.raises(TypeError, match='Unrecognized sweepable'):
        for _ in cirq.study.to_resolvers('nope'):
            pass


def test_to_sweeps_none():
    assert cirq.study.to_sweeps(None) == [cirq.UnitSweep]


def test_to_sweeps_single():
    resolver = cirq.ParamResolver({})
    assert cirq.study.to_sweeps(resolver) == [cirq.UnitSweep]
    assert cirq.study.to_sweeps({}) == [cirq.UnitSweep]


def test_to_sweeps_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.study.to_sweeps(sweep) == [sweep]


def test_to_sweeps_iterable():
    resolvers = [cirq.ParamResolver({'a': 2}), cirq.ParamResolver({'a': 1})]
    sweeps = [
        cirq.study.Zip(cirq.Points('a', [2])),
        cirq.study.Zip(cirq.Points('a', [1])),
    ]
    assert cirq.study.to_sweeps(resolvers) == sweeps
    assert cirq.study.to_sweeps([{'a': 2}, {'a': 1}]) == sweeps


def test_to_sweeps_iterable_sweeps():
    sweeps = [cirq.Linspace('a', 0, 1, 10), cirq.Linspace('b', 0, 1, 10)]
    assert cirq.study.to_sweeps(sweeps) == sweeps


def test_to_sweeps_dictionary_of_list():
    with pytest.warns(DeprecationWarning, match='dict_to_product_sweep'):
        assert cirq.study.to_sweeps({'t': [0, 2, 3]}) == cirq.study.to_sweeps(
            [{'t': 0}, {'t': 2}, {'t': 3}]
        )
        assert cirq.study.to_sweeps({'t': [0, 1], 's': [2, 3], 'r': 4}) == cirq.study.to_sweeps(
            [
                {'t': 0, 's': 2, 'r': 4},
                {'t': 0, 's': 3, 'r': 4},
                {'t': 1, 's': 2, 'r': 4},
                {'t': 1, 's': 3, 'r': 4},
            ]
        )


def test_to_sweeps_invalid():
    with pytest.raises(TypeError, match='Unrecognized sweepable'):
        cirq.study.to_sweeps('nope')


def test_to_sweep_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.to_sweep(sweep) is sweep


@pytest.mark.parametrize(
    'r_gen',
    [
        lambda: {'a': 1},
        lambda: {sympy.Symbol('a'): 1},
        lambda: cirq.ParamResolver({'a': 1}),
        lambda: cirq.ParamResolver({sympy.Symbol('a'): 1}),
    ],
)
def test_to_sweep_single_resolver(r_gen):
    sweep = cirq.to_sweep(r_gen())
    assert isinstance(sweep, cirq.Sweep)
    assert list(sweep) == [cirq.ParamResolver({'a': 1})]


@pytest.mark.parametrize(
    'r_list_gen',
    [
        # Lists
        lambda: [{'a': 1}, {'a': 1.5}],
        lambda: [{sympy.Symbol('a'): 1}, {sympy.Symbol('a'): 1.5}],
        lambda: [cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 1.5})],
        lambda: [
            cirq.ParamResolver({sympy.Symbol('a'): 1}),
            cirq.ParamResolver({sympy.Symbol('a'): 1.5}),
        ],
        lambda: [{'a': 1}, cirq.ParamResolver({sympy.Symbol('a'): 1.5})],
        lambda: ({'a': 1}, {'a': 1.5}),
        # Iterators
        lambda: (r for r in [{'a': 1}, {'a': 1.5}]),
        lambda: {object(): r for r in [{'a': 1}, {'a': 1.5}]}.values(),
    ],
)
def test_to_sweep_resolver_list(r_list_gen):
    sweep = cirq.to_sweep(r_list_gen())
    assert isinstance(sweep, cirq.Sweep)
    assert list(sweep) == [cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 1.5})]


def test_to_sweep_type_error():
    with pytest.raises(TypeError, match='Unexpected sweep'):
        cirq.to_sweep(5)
