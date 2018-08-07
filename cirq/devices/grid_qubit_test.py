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
"""Tests for grid_qubit."""

import pytest

import cirq


def test_xmon_qubit_init():
    q = cirq.GridQubit(3, 4)
    assert q.row == 3
    assert q.col == 4


def test_xmon_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GridQubit(0, 0))
    eq.make_equality_group(lambda: cirq.GridQubit(1, 0))
    eq.make_equality_group(lambda: cirq.GridQubit(0, 1))
    eq.make_equality_group(lambda: cirq.GridQubit(50, 25))


def test_xmon_qubit_ordering():
    assert cirq.GridQubit(0, 0) < cirq.GridQubit(0, 1)
    assert cirq.GridQubit(0, 0) < cirq.GridQubit(1, 0)
    assert cirq.GridQubit(0, 0) < cirq.GridQubit(1, 1)
    assert cirq.GridQubit(0, 0) <= cirq.GridQubit(0, 0)
    assert cirq.GridQubit(0, 0) <= cirq.GridQubit(0, 1)
    assert cirq.GridQubit(0, 0) <= cirq.GridQubit(1, 0)
    assert cirq.GridQubit(0, 0) <= cirq.GridQubit(1, 1)

    assert cirq.GridQubit(1, 1) > cirq.GridQubit(0, 1)
    assert cirq.GridQubit(1, 1) > cirq.GridQubit(1, 0)
    assert cirq.GridQubit(1, 1) > cirq.GridQubit(0, 0)
    assert cirq.GridQubit(1, 1) >= cirq.GridQubit(1, 1)
    assert cirq.GridQubit(1, 1) >= cirq.GridQubit(0, 1)
    assert cirq.GridQubit(1, 1) >= cirq.GridQubit(1, 0)
    assert cirq.GridQubit(1, 1) >= cirq.GridQubit(0, 0)


def test_xmon_qubit_is_adjacent():
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(0, 1))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(0, -1))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(1, 0))
    assert cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, 0))

    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(+1, -1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(+1, +1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, -1))
    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(-1, +1))

    assert not cirq.GridQubit(0, 0).is_adjacent(cirq.GridQubit(2, 0))

    assert cirq.GridQubit(500, 999).is_adjacent(cirq.GridQubit(501, 999))
    assert not cirq.GridQubit(500, 999).is_adjacent(cirq.GridQubit(5034, 999))


def test_to_proto():
    q = cirq.GridQubit(5, 6)

    # Create a new message.
    proto = q.to_proto_dict()
    assert proto == {'row': 5, 'col': 6}


def test_from_proto():
    q = cirq.GridQubit(5, 6)
    q2 = cirq.GridQubit.from_proto_dict(q.to_proto_dict())
    assert q2 == q


def test_from_proto_bad_dict():
    with pytest.raises(ValueError):
        cirq.GridQubit.from_proto_dict({'row': 1})
    with pytest.raises(ValueError):
        cirq.GridQubit.from_proto_dict({'col': 1})
    with pytest.raises(ValueError):
        cirq.GridQubit.from_proto_dict({})
    with pytest.raises(ValueError):
        cirq.GridQubit.from_proto_dict({'nothing': 1})
