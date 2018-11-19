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


@cirq.class_equality
class BasicC:
    pass


@cirq.class_equality
class BasicD:
    pass


class BasicE:
    pass


class BasicCa(BasicC):
    pass


class BasicCb(BasicC):
    pass


def test_class_equality_basic():

    # Class equality works as expected.
    assert cirq.class_equals(BasicC(), BasicC()) is True
    assert cirq.class_equals(BasicC(), BasicCa()) is True
    assert cirq.class_equals(BasicCa(), BasicC()) is True
    assert cirq.class_equals(BasicCa(), BasicCa()) is True
    assert cirq.class_equals(BasicCa(), BasicCb()) is True
    assert cirq.class_equals(BasicC(), BasicD()) is False
    assert cirq.class_equals(BasicC(), BasicE()) is NotImplemented
    assert cirq.class_equals(BasicE(), BasicC()) is NotImplemented


@cirq.class_equality(distinct_child_types=True)
class DistinctC:
    pass


@cirq.class_equality(distinct_child_types=True)
class DistinctD:
    pass


class DistinctCa(DistinctC):
    pass


class DistinctCb(DistinctC):
    pass


def test_value_equality_distinct_child_types():

    # Distinct class equality works as expected.
    assert cirq.class_equals(DistinctC(), DistinctC()) is True
    assert cirq.class_equals(DistinctC(), DistinctCa()) is False
    assert cirq.class_equals(DistinctCa(), DistinctC()) is False
    assert cirq.class_equals(DistinctCa(), DistinctCa()) is True
    assert cirq.class_equals(DistinctCa(), DistinctCb()) is False
    assert cirq.class_equals(DistinctC(), DistinctD()) is False

