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

import cirq


@cirq.value_equality
class BasicC:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


@cirq.value_equality
class BasicD:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


@cirq.value_equality(manual_cls=True)
class MasqueradePositiveD:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x

    def _value_equality_values_cls_(self):
        return BasicD if self.x > 0 else MasqueradePositiveD


class BasicCa(BasicC):
    pass


class BasicCb(BasicC):
    pass


def test_value_equality_basic():

    # Lookup works across equivalent types.
    v = {BasicC(1): 4, BasicCa(2): 5}
    assert v[BasicCa(1)] == v[BasicC(1)] == 4
    assert v[BasicCa(2)] == 5

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(BasicC(1), BasicC(1), BasicCa(1), BasicCb(1))
    eq.add_equality_group(BasicD(1))
    eq.add_equality_group(BasicC(2))
    eq.add_equality_group(BasicCa(3))


def test_value_equality_manual():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(MasqueradePositiveD(3), BasicD(3))
    eq.add_equality_group(MasqueradePositiveD(4), MasqueradePositiveD(4), BasicD(4))
    eq.add_equality_group(MasqueradePositiveD(-1), MasqueradePositiveD(-1))
    eq.add_equality_group(BasicD(-1))


@cirq.value_equality(unhashable=True)
class UnhashableC:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


@cirq.value_equality(unhashable=True)
class UnhashableD:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


class UnhashableCa(UnhashableC):
    pass


class UnhashableCb(UnhashableC):
    pass


def test_value_equality_unhashable():
    # Not possible to use as a dictionary key.
    with pytest.raises(TypeError, match='unhashable'):
        _ = {UnhashableC(1): 4}

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(UnhashableC(1), UnhashableC(1), UnhashableCa(1), UnhashableCb(1))
    eq.add_equality_group(UnhashableC(2))
    eq.add_equality_group(UnhashableD(1))


@cirq.value_equality(distinct_child_types=True)
class DistinctC:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


@cirq.value_equality(distinct_child_types=True)
class DistinctD:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


class DistinctCa(DistinctC):
    pass


class DistinctCb(DistinctC):
    pass


def test_value_equality_distinct_child_types():
    # Lookup is distinct across child types.
    v = {DistinctC(1): 4, DistinctCa(1): 5, DistinctCb(1): 6}
    assert v[DistinctC(1)] == 4
    assert v[DistinctCa(1)] == 5
    assert v[DistinctCb(1)] == 6

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(DistinctC(1), DistinctC(1))
    eq.add_equality_group(DistinctCa(1), DistinctCa(1))
    eq.add_equality_group(DistinctCb(1), DistinctCb(1))
    eq.add_equality_group(DistinctC(2))
    eq.add_equality_group(DistinctD(1))


@cirq.value_equality(approximate=True)
class ApproxE:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


def test_value_equality_approximate():
    assert cirq.approx_eq(ApproxE(0.0), ApproxE(0.0), atol=0.1)
    assert cirq.approx_eq(ApproxE(0.0), ApproxE(0.2), atol=0.3)
    assert not cirq.approx_eq(ApproxE(0.0), ApproxE(0.2), atol=0.1)


@cirq.value_equality(approximate=True)
class PeriodicF:
    def __init__(self, x, n):
        self.x = x
        self.n = n

    def _value_equality_values_(self):
        return self.x

    def _value_equality_approximate_values_(self):
        return self.x % self.n


def test_value_equality_approximate_specialized():
    assert PeriodicF(1, 4) != PeriodicF(5, 4)
    assert cirq.approx_eq(PeriodicF(1, 4), PeriodicF(5, 4), atol=0.1)
    assert not cirq.approx_eq(PeriodicF(1, 4), PeriodicF(6, 4), atol=0.1)


def test_value_equality_approximate_not_supported():
    assert not cirq.approx_eq(BasicC(0.0), BasicC(0.1), atol=0.2)


class ApproxEa(ApproxE):
    pass


class ApproxEb(ApproxE):
    pass


@cirq.value_equality(distinct_child_types=True, approximate=True)
class ApproxG:
    def __init__(self, x):
        self.x = x

    def _value_equality_values_(self):
        return self.x


class ApproxGa(ApproxG):
    pass


class ApproxGb(ApproxG):
    pass


def test_value_equality_approximate_typing():
    assert not cirq.approx_eq(ApproxE(0.0), PeriodicF(0.0, 1.0), atol=0.1)
    assert cirq.approx_eq(ApproxEa(0.0), ApproxEb(0.0), atol=0.1)
    assert cirq.approx_eq(ApproxG(0.0), ApproxG(0.0), atol=0.1)
    assert not cirq.approx_eq(ApproxGa(0.0), ApproxGb(0.0), atol=0.1)
    assert not cirq.approx_eq(ApproxG(0.0), ApproxGb(0.0), atol=0.1)


def test_value_equality_forgot_method():
    with pytest.raises(TypeError, match='_value_equality_values_'):

        @cirq.value_equality
        class _:
            pass


def test_bad_manual_cls_incompatible_args():
    with pytest.raises(ValueError, match='incompatible'):

        @cirq.value_equality(manual_cls=True, distinct_child_types=True)
        class _:
            pass


def test_bad_manual_cls_forgot_method():
    with pytest.raises(TypeError, match='_value_equality_values_cls_'):

        @cirq.value_equality(manual_cls=True)
        class _:
            def _value_equality_values_(self):
                pass
