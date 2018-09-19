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

import fractions
import pytest

from cirq.testing.order_tester import OrderTester

class UnorderableClass:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        return isinstance(other, UnorderableClass)

    def __ne__(self, other):
        return not isinstance(other, UnorderableClass)

    def __lt__(self, other):
        return NotImplemented

    def __hash__(self):
        return hash(UnorderableClass)

def test_add_ordering_group_correct():
    ot = OrderTester()
    ot.add_ascending(-4, 0)
    ot.add_ascending(1, 2)
    ot.add_ascending_equivalence_group(fractions.Fraction(6, 2),
                                       fractions.Fraction(12, 4), 3, 3.0)
    ot.add_ascending_equivalence_group(float('inf'), float('inf'))

def test_add_ordering_group_incorrect():
    ot = OrderTester()
    ot.add_ascending(0)
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(0, 0)
    ot.add_ascending(1, 2)
    with pytest.raises(AssertionError):
        ot.add_ascending(object, object)  # not ascending within call
    with pytest.raises(AssertionError):
        ot.add_ascending(1, 3)  # not ascending w.r.t. previous call
    with pytest.raises(AssertionError):
        ot.add_ascending(6, 6)  # not ascending within call
    with pytest.raises(AssertionError):
        ot.add_ascending(99, 10)  # not ascending within call

def test_add_ordering_equivalence_group_incorrect():
    ot = OrderTester()
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(1, 3)  # not an equivalence group
    ot.add_ascending(1)
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(0, 0.)  # not ascending w.r.t
                                                   # previous items
    with pytest.raises(AssertionError):
        ot.add_ascending(UnorderableClass())
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(UnorderableClass(),
            UnorderableClass())

def test_add_ordering_equivalence_group_bad_hash():
    class KeyHash:
        def __init__(self, k, h):
            self._k = k
            self._h = h

        def __eq__(self, other):
            return isinstance(other, KeyHash) and self._k == other._k

        def __ne__(self, other):
            return not self == other

        def __lt__(self, other):
            return isinstance(other, KeyHash) and self._k < other._k

        def __hash__(self):
            return self._h


    ot = OrderTester()
    ot.add_ascending_equivalence_group(KeyHash('a', 5), KeyHash('a', 5))
    ot.add_ascending_equivalence_group(KeyHash('b', 5))
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(KeyHash('c', 2), KeyHash('c', 3))
