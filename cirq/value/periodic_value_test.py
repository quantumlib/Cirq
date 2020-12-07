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


def test_periodic_value_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PeriodicValue(1, 2),
        cirq.PeriodicValue(1, 2),
        cirq.PeriodicValue(3, 2),
        cirq.PeriodicValue(3, 2),
        cirq.PeriodicValue(5, 2),
        cirq.PeriodicValue(-1, 2),
    )
    eq.add_equality_group(
        cirq.PeriodicValue(1.5, 2.0),
        cirq.PeriodicValue(1.5, 2.0),
    )
    eq.add_equality_group(cirq.PeriodicValue(0, 2))
    eq.add_equality_group(cirq.PeriodicValue(1, 3))
    eq.add_equality_group(cirq.PeriodicValue(2, 4))


def test_periodic_value_approx_eq_basic():
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.0), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.0), atol=0.1)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.2), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.2), atol=0.1)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.2), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.2), atol=0.1)


def test_periodic_value_approx_eq_normalized():
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 3.0), cirq.PeriodicValue(4.1, 3.0), atol=0.2)
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 3.0), cirq.PeriodicValue(-2.1, 3.0), atol=0.2)


def test_periodic_value_approx_eq_boundary():
    assert cirq.approx_eq(cirq.PeriodicValue(0.0, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.2)
    assert cirq.approx_eq(cirq.PeriodicValue(0.1, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.3)
    assert cirq.approx_eq(cirq.PeriodicValue(1.9, 2.0), cirq.PeriodicValue(0.1, 2.0), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(0.1, 2.0), cirq.PeriodicValue(1.9, 2.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(0, 1.0), cirq.PeriodicValue(0.5, 1.0), atol=0.6)
    assert not cirq.approx_eq(cirq.PeriodicValue(0, 1.0), cirq.PeriodicValue(0.5, 1.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(0.4, 1.0), cirq.PeriodicValue(0.6, 1.0), atol=0.3)


def test_periodic_value_types_mismatch():
    assert not cirq.approx_eq(cirq.PeriodicValue(0.0, 2.0), 0.0, atol=0.2)
    assert not cirq.approx_eq(0.0, cirq.PeriodicValue(0.0, 2.0), atol=0.2)


@pytest.mark.parametrize(
    'value, is_parameterized, parameter_names',
    [
        (cirq.PeriodicValue(1.0, 3.0), False, set()),
        (cirq.PeriodicValue(0.0, sympy.Symbol('p')), True, {'p'}),
        (cirq.PeriodicValue(sympy.Symbol('v'), 3.0), True, {'v'}),
        (cirq.PeriodicValue(sympy.Symbol('v'), sympy.Symbol('p')), True, {'p', 'v'}),
    ],
)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_periodic_value_is_parameterized(value, is_parameterized, parameter_names, resolve_fn):
    assert cirq.is_parameterized(value) == is_parameterized
    assert cirq.parameter_names(value) == parameter_names
    resolved = resolve_fn(value, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)


@pytest.mark.parametrize(
    'val',
    [
        cirq.PeriodicValue(0.4, 1.0),
        cirq.PeriodicValue(0.0, 2.0),
        cirq.PeriodicValue(1.0, 3),
        cirq.PeriodicValue(-2.1, 3.0),
        cirq.PeriodicValue(sympy.Symbol('v'), sympy.Symbol('p')),
        cirq.PeriodicValue(2.0, sympy.Symbol('p')),
        cirq.PeriodicValue(sympy.Symbol('v'), 3),
    ],
)
def test_periodic_value_repr(val):
    cirq.testing.assert_equivalent_repr(val)
