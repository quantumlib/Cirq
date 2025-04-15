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

from typing import Iterable

import cirq
from cirq_google.line.placement.chip import above, below, chip_as_adjacency_list, left_of, right_of


def test_neighbours():
    qubit = cirq.GridQubit(0, 0)
    assert above(qubit) == cirq.GridQubit(0, -1)
    assert below(qubit) == cirq.GridQubit(0, 1)
    assert right_of(qubit) == cirq.GridQubit(1, 0)
    assert left_of(qubit) == cirq.GridQubit(-1, 0)


def test_qubit_not_mutated():
    qubit = cirq.GridQubit(0, 0)

    above(qubit)
    assert qubit == cirq.GridQubit(0, 0)

    below(qubit)
    assert qubit == cirq.GridQubit(0, 0)

    right_of(qubit)
    assert qubit == cirq.GridQubit(0, 0)

    left_of(qubit)
    assert qubit == cirq.GridQubit(0, 0)


class FakeDevice(cirq.Device):
    def __init__(self, qubits):
        self.qubits = qubits

    @property
    def metadata(self):
        return cirq.DeviceMetadata(self.qubits, None)


def _create_device(qubits: Iterable[cirq.GridQubit]):
    return FakeDevice(qubits)


def test_empty():
    assert chip_as_adjacency_list(_create_device([])) == {}


def test_single_qubit():
    q00 = cirq.GridQubit(0, 0)
    assert chip_as_adjacency_list(_create_device([q00])) == {q00: []}


def test_two_close_qubits():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    assert chip_as_adjacency_list(_create_device([q00, q01])) == {q00: [q01], q01: [q00]}


def test_two_qubits_apart():
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    assert chip_as_adjacency_list(_create_device([q00, q11])) == {q00: [], q11: []}


def test_three_qubits_in_row():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    assert chip_as_adjacency_list(_create_device([q00, q01, q02])) == {
        q00: [q01],
        q01: [q00, q02],
        q02: [q01],
    }


def test_square_of_four():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    assert chip_as_adjacency_list(_create_device([q00, q01, q10, q11])) == {
        q00: [q01, q10],
        q01: [q00, q11],
        q10: [q00, q11],
        q11: [q10, q01],
    }
