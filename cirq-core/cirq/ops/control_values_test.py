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

from cirq.ops import control_values as cv
import cirq


def test_empty_init():
    assert [cv.flatten(p) for p in cv.ControlValues([])] == [()]


def test_init_control_values():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cv.ControlValues([]))
    tests = [
        ([1], [(1,)]),
        ([[0, 1], 1], [[(0, 1), (1, 1)]]),
        ([[[0, 1], [1, 0]]], [[(0, 1), (1, 0)]]),
    ]
    for control_values, want in tests:
        eq.add_equality_group(cv.ControlValues(control_values), want)

    with pytest.raises(TypeError):
        _ = cv.ControlValues([1.0])


def test_copy_constructor():
    eq = cirq.testing.EqualsTester()
    tests = [[], [1], [[0, 1], 1], [[[0, 1], [1, 0]]]]
    for control_values in tests:
        values = [cv.ControlValues([cv.ControlValues([val])]) for val in control_values]
        eq.add_equality_group(
            cv.ControlValues([cv.ControlValues(control_values)]), cv.ControlValues(values)
        )


def test_constrained_init():
    eq = cirq.testing.EqualsTester()
    tests = [
        ([[0, 1], [1, 0]], [(0, 1), (1, 0)]),
        ([[0, 0], [0, 1]], [(0, 0), (0, 1)]),
        ([[1, 0], [1, 1]], [(0, 1), (1, 1)]),
    ]
    for control_values, want in tests:
        control_vals = cv.ConstrainedVars(control_values)
        got = [cv.flatten(product) for product in control_vals]
        eq.add_equality_group(sorted(got), want)


def test_and():
    eq = cirq.testing.EqualsTester()
    originals = [[1], [[0, 1], 1], [[[0, 1], [1, 0]]]]
    for control_values1 in originals:
        for control_values2 in originals:
            control_vals1 = cv.ControlValues(control_values1)
            control_vals2 = cv.ControlValues(control_values2)
            want = [[v1 + v2 for v1 in control_vals1 for v2 in control_vals2]]
            control_vals1.product(control_vals2)
            eq.add_equality_group(control_vals1, want)


def test_slicing_not_supported():
    control_vals = cv.ControlValues([[[0, 1], [1, 0]]])
    with pytest.raises(ValueError):
        _ = control_vals[0:1]


def test_check_dimentionality():
    empty_control_vals = cv.ControlValues([])
    empty_control_vals.check_dimentionality()

    control_values = cv.ControlValues([[0, 1], 1])
    with pytest.raises(ValueError):
        control_values.check_dimentionality()


def test_pop():
    tests = [
        ([[0, 1], 1], [(0,), (1,)]),
        ([[[0, 1], [1, 0]], 0, 1], [(0, 1, 0), (1, 0, 0)]),
    ]
    for control_values, want in tests:
        control_vals = cv.ControlValues(control_values)
        control_vals.pop()
        got = [cv.flatten(product) for product in control_vals]
        assert want == sorted(got)


def test_arrangements():
    tests = [
        ((), [()]),
        ([1], [(1,)]),
        ([(0, 1), (1,)], [(0, 1), (1,)]),
        ([((0, 1), (1, 0))], [((0, 1), (1, 0))]),
    ]
    for control_values, want in tests:
        control_vals = cv.ControlValues(control_values)
        assert want == control_vals.arrangements()
