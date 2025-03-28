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

import cirq

CMP_OPS = [
    lambda a, b: a == b,
    lambda a, b: a != b,
    lambda a, b: a < b,
    lambda a, b: a > b,
    lambda a, b: a <= b,
    lambda a, b: a >= b,
]


class MockValue:
    def __init__(self, val, eq, ne, lt, gt, le, ge):
        self.val = val
        self._eq = eq
        self._ne = ne
        self._lt = lt
        self._gt = gt
        self._le = le
        self._ge = ge

    __hash__ = None  # type: ignore

    def __eq__(self, other):
        return self._eq(self, other)

    def __ne__(self, other):
        return self._ne(self, other)

    def __lt__(self, other):
        return self._lt(self, other)

    def __gt__(self, other):
        return self._gt(self, other)

    def __le__(self, other):
        return self._le(self, other)

    def __ge__(self, other):
        return self._ge(self, other)

    def __repr__(self):
        return f'MockValue(val={self.val!r}, ...)'


def test_add_ordering_group_correct():
    ot = cirq.testing.OrderTester()
    ot.add_ascending(-4, 0)
    ot.add_ascending(1, 2)
    ot.add_ascending_equivalence_group(fractions.Fraction(6, 2), fractions.Fraction(12, 4), 3, 3.0)
    ot.add_ascending_equivalence_group(float('inf'), float('inf'))


def test_add_ordering_group_incorrect():
    ot = cirq.testing.OrderTester()
    ot.add_ascending(0)
    with pytest.raises(AssertionError):
        ot.add_ascending_equivalence_group(0, 0)
    ot.add_ascending(1, 2)
    with pytest.raises(AssertionError):
        ot.add_ascending(20, 20)  # not ascending within call
    with pytest.raises(AssertionError):
        ot.add_ascending(1, 3)  # not ascending w.r.t. previous call
    with pytest.raises(AssertionError):
        ot.add_ascending(6, 6)  # not ascending within call
    with pytest.raises(AssertionError):
        ot.add_ascending(99, 10)  # not ascending within call
    with pytest.raises(AssertionError):
        ot.add_ascending(0)


def test_propagates_internal_errors():
    class UnorderableClass:  # pragma: no cover
        def __eq__(self, other):
            return NotImplemented

        def __ne__(self, other):
            return NotImplemented

        def __lt__(self, other):
            raise ValueError('oh no')

        def __le__(self, other):
            return NotImplemented

        def __ge__(self, other):
            return NotImplemented

        def __gt__(self, other):
            return NotImplemented

    ot = cirq.testing.OrderTester()
    with pytest.raises(ValueError, match='oh no'):
        ot.add_ascending(UnorderableClass())


def test_add_ascending_equivalence_group():
    ot = cirq.testing.OrderTester()
    with pytest.raises(AssertionError, match='Expected X=1 to equal Y=3'):
        ot.add_ascending_equivalence_group(1, 3)

    ot.add_ascending_equivalence_group(2)
    ot.add_ascending_equivalence_group(4)

    with pytest.raises(AssertionError, match='Expected X=4 to be less than Y=3'):
        ot.add_ascending_equivalence_group(3)

    ot.add_ascending_equivalence_group(5)


def test_fails_to_return_not_implemented_vs_unknown():
    def make_impls(bad_index: int, bad_result: bool):
        def make_impl(i, op):
            def impl(x, y):
                if isinstance(y, MockValue):
                    return op(x.val, y.val)
                if bad_index == i:
                    return bad_result
                return NotImplemented

            return impl

        return [make_impl(i, op) for i, op in enumerate(CMP_OPS)]

    ot = cirq.testing.OrderTester()
    for k in range(len(CMP_OPS)):
        for b in [False, True]:
            item = MockValue(0, *make_impls(bad_index=k, bad_result=b))
            with pytest.raises(AssertionError, match='return NotImplemented'):
                ot.add_ascending(item)

    good_impls = make_impls(bad_index=-1, bad_result=NotImplemented)
    ot.add_ascending(MockValue(0, *good_impls))
    ot.add_ascending(MockValue(1, *good_impls))


def test_fails_on_inconsistent_hashes():
    class ModifiedHash(tuple):
        def __hash__(self):
            return super().__hash__() ^ 1

    ot = cirq.testing.OrderTester()
    ot.add_ascending((1, 0), (1, 1))
    ot.add_ascending(ModifiedHash((1, 2)), ModifiedHash((2, 0)))
    ot.add_ascending_equivalence_group((2, 2), (2, 2))
    ot.add_ascending_equivalence_group(ModifiedHash((3, 3)), ModifiedHash((3, 3)))

    with pytest.raises(AssertionError, match='different hashes'):
        ot.add_ascending_equivalence_group((4, 4), ModifiedHash((4, 4)))
