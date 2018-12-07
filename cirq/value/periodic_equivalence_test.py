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

import cirq


def test_periodic_equivalence_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PeriodicEquivalence(1, 2),
        cirq.PeriodicEquivalence(1, 2),
        cirq.PeriodicEquivalence(3, 2),
        cirq.PeriodicEquivalence(3, 2),
        cirq.PeriodicEquivalence(5, 2),
        cirq.PeriodicEquivalence(-1, 2)
    )
    eq.add_equality_group(
        cirq.PeriodicEquivalence(1.5, 2.0),
        cirq.PeriodicEquivalence(1.5, 2.0),
    )
    eq.add_equality_group(cirq.PeriodicEquivalence(0, 2))
    eq.add_equality_group(cirq.PeriodicEquivalence(1, 3))
    eq.add_equality_group(cirq.PeriodicEquivalence(2, 4))


def test_periodic_equivalence_approx_eq_basic():
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.0, 2.0),
        atol=0.1
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.2, 2.0),
        atol=0.3
    )
    assert not cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.2, 2.0),
        atol=0.1
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.0, 2.2),
        atol=0.3
    )
    assert not cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.0, 2.2),
        atol=0.1
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.2, 2.2),
        atol=0.3
    )
    assert not cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 2.0),
        cirq.PeriodicEquivalence(1.2, 2.2),
        atol=0.1
    )


def test_periodic_equivalence_approx_eq_normalized():
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 3.0),
        cirq.PeriodicEquivalence(4.1, 3.0),
        atol=0.2
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.0, 3.0),
        cirq.PeriodicEquivalence(-2.1, 3.0),
        atol=0.2
    )


def test_periodic_equivalence_approx_eq_boundary():
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(0.0, 2.0),
        cirq.PeriodicEquivalence(1.9, 2.0),
        atol=0.2
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(0.1, 2.0),
        cirq.PeriodicEquivalence(1.9, 2.0),
        atol=0.3
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(1.9, 2.0),
        cirq.PeriodicEquivalence(0.1, 2.0),
        atol=0.3
    )
    assert not cirq.approx_eq(
        cirq.PeriodicEquivalence(0.1, 2.0),
        cirq.PeriodicEquivalence(1.9, 2.0),
        atol=0.1
    )
    assert cirq.approx_eq(
        cirq.PeriodicEquivalence(0, 1.0),
        cirq.PeriodicEquivalence(0.5, 1.0),
        atol=0.6
    )


def test_periodic_equivalence_types_mismatch():
    assert not cirq.approx_eq(cirq.PeriodicEquivalence(0.0, 2.0), 0.0, atol=0.2)
    assert not cirq.approx_eq(0.0, cirq.PeriodicEquivalence(0.0, 2.0), atol=0.2)
