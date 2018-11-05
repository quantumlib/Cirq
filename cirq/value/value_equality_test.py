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


def test_value_equality_basic():
    @cirq.value_equality
    class C:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    @cirq.value_equality
    class D:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    class Ca(C):
        pass

    class Cb(C):
        pass

    # Lookup works across equivalent types.
    v = {C(1): 4, Ca(2): 5}
    assert v[Ca(1)] == v[C(1)] == 4
    assert v[Ca(2)] == 5

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(C(1), C(1), Ca(1), Cb(1))
    eq.add_equality_group(D(1))
    eq.add_equality_group(C(2))
    eq.add_equality_group(Ca(3))


def test_value_equality_unhashable():
    @cirq.value_equality(unhashable=True)
    class C:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    @cirq.value_equality(unhashable=True)
    class D:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    class Ca(C):
        pass

    class Cb(C):
        pass

    # Not possible to use as a dictionary key.
    with pytest.raises(TypeError, match='unhashable'):
        _ = {C(1): 4}

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(C(1), C(1), Ca(1), Cb(1))
    eq.add_equality_group(C(2))
    eq.add_equality_group(D(1))


def test_value_equality_distinct_child_types():
    @cirq.value_equality(distinct_child_types=True)
    class C:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    @cirq.value_equality(distinct_child_types=True)
    class D:
        def __init__(self, x):
            self.x = x

        def _value_equality_values_(self):
            return self.x

    class Ca(C):
        pass

    class Cb(C):
        pass

    # Lookup is distinct across child types.
    v = {C(1): 4, Ca(1): 5, Cb(1): 6}
    assert v[C(1)] == 4
    assert v[Ca(1)] == 5
    assert v[Cb(1)] == 6

    # Equality works as expected.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(C(1), C(1))
    eq.add_equality_group(Ca(1), Ca(1))
    eq.add_equality_group(Cb(1), Cb(1))
    eq.add_equality_group(C(2))
    eq.add_equality_group(D(1))
