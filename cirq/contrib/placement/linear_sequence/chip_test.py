# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

from cirq.contrib.placement.linear_sequence.chip import chip_as_adjacency_list, \
    yx_cmp, above, below, right_of, left_of
from cirq.google import XmonDevice, XmonQubit
from cirq.value import Duration


def test_neighbours():
    qubit = XmonQubit(0, 0)
    assert above(qubit) == XmonQubit(0, -1)
    assert below(qubit) == XmonQubit(0, 1)
    assert right_of(qubit) == XmonQubit(1, 0)
    assert left_of(qubit) == XmonQubit(-1, 0)


def test_qubit_not_mutated():
    qubit = XmonQubit(0, 0)

    above(qubit)
    assert qubit == XmonQubit(0, 0)

    below(qubit)
    assert qubit == XmonQubit(0, 0)

    right_of(qubit)
    assert qubit == XmonQubit(0, 0)

    left_of(qubit)
    assert qubit == XmonQubit(0, 0)


def test_lower():
    assert yx_cmp(XmonQubit(0, 0), XmonQubit(1, 1)) < 0
    assert yx_cmp(XmonQubit(0, 1), XmonQubit(1, 1)) < 0
    assert yx_cmp(XmonQubit(1, 0), XmonQubit(0, 1)) < 0


def test_equal():
    assert yx_cmp(XmonQubit(0, 0), XmonQubit(0, 0)) == 0
    assert yx_cmp(XmonQubit(1, 1), XmonQubit(1, 1)) == 0


def test_greater():
    assert yx_cmp(XmonQubit(1, 1), XmonQubit(0, 0)) > 0
    assert yx_cmp(XmonQubit(1, 1), XmonQubit(0, 1)) > 0
    assert yx_cmp(XmonQubit(0, 1), XmonQubit(1, 0)) > 0


def _create_device(qubits: Iterable[XmonQubit]) -> XmonDevice:
    return XmonDevice(Duration(nanos=0), Duration(nanos=0), Duration(nanos=0),
                      qubits)


def test_empty():
    assert chip_as_adjacency_list(_create_device([])) == {}


def test_single_qubit():
    q00 = XmonQubit(0, 0)
    assert chip_as_adjacency_list(_create_device([q00])) == {q00: []}


def test_two_close_qubits():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    assert chip_as_adjacency_list(_create_device([q00, q01])) == {q00: [q01],
                                                                  q01: [q00]}


def test_two_qubits_apart():
    q00 = XmonQubit(0, 0)
    q11 = XmonQubit(1, 1)
    assert chip_as_adjacency_list(_create_device([q00, q11])) == {q00: [],
                                                                  q11: []}


def test_three_qubits_in_row():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    assert chip_as_adjacency_list(_create_device([q00, q01, q02])) == {
        q00: [q01], q01: [q00, q02], q02: [q01]}


def test_square_of_four():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q10 = XmonQubit(1, 0)
    q11 = XmonQubit(1, 1)
    assert chip_as_adjacency_list(_create_device([q00, q01, q10, q11])) == {
        q00: [q01, q10], q01: [q00, q11], q10: [q00, q11], q11: [q10, q01]}
