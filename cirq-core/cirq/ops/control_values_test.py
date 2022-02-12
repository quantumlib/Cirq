# Copyright 2021 The Cirq Developers
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

import cirq


flatten = cirq.ops.ControlValues.flatten
builder = lambda values: cirq.ops.ControlValuesBuilder().append(values).build()


def test_empty_init():
    assert [flatten(p) for p in builder([])] == [()]


def test_init_control_values():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(builder([]))
    tests = [
        ([1], [(1,)]),
        ([[0, 1], 1], [(0, 1), (1, 1)]),
        ([[[0, 1], [1, 0]]], [(0, 1), (1, 0)]),
    ]
    for control_values, want in tests:
        got = [flatten(c) for c in builder(control_values)]
        eq.add_equality_group(got, want)

    with pytest.raises(TypeError):
        _ = builder([1.0])


def test_constrained_init():
    eq = cirq.testing.EqualsTester()
    tests = [
        ([[0, 1], [1, 0]], [(0, 1), (1, 0)]),
        ([[0, 0], [0, 1]], [(0, 0), (0, 1)]),
        ([[1, 0], [1, 1]], [(0, 1), (1, 1)]),
    ]
    for control_values, want in tests:
        control_vals = cirq.ops.ConstrainedValues.factory(control_values)
        got = [flatten(product) for product in control_vals]
        eq.add_equality_group(sorted(got), want)


def test_product():
    eq = cirq.testing.EqualsTester()
    originals = [[1], [[0, 1], 1], [[[0, 1], [1, 0]]]]
    for control_values1 in originals:
        for control_values2 in originals:
            control_vals1 = builder(control_values1)
            control_vals2 = builder(control_values2)
            want = [[flatten(v1 + v2) for v1 in control_vals1 for v2 in control_vals2]]
            got = [[flatten(c) for c in control_vals1.product(control_vals2)]]
            eq.add_equality_group(got, want)


def test_slicing_not_supported():
    control_vals = builder([[[0, 1], [1, 0]]])
    with pytest.raises(ValueError):
        _ = control_vals[0:1]


def test_check_dimensionality():
    empty_control_vals = builder([])
    empty_control_vals.check_dimensionality()

    control_values = builder([[0, 1], 1])
    with pytest.raises(ValueError):
        control_values.check_dimensionality()


def test_pop():
    tests = [
        ([[0, 1], 1], [(0,), (1,)]),
        ([[[0, 1], [1, 0]], 0, 1], [(0, 1, 0), (1, 0, 0)]),
    ]
    for control_values, want in tests:
        control_vals = builder(control_values)
        control_vals = control_vals.pop()
        got = [flatten(product) for product in control_vals]
        assert want == sorted(got)


def test_arrangements():
    tests = [
        ((), ((),)),
        ([1], ((1,),)),
        ([(0, 1), (1,)], ((0, 1), (1,))),
        ([((0, 1), (1, 0))], (((0, 1), (1, 0)),)),
    ]
    for control_values, want in tests:
        control_vals = builder(control_values)
        assert want == control_vals.arrangements()
