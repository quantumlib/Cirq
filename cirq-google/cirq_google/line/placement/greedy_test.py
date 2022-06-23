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
from unittest import mock
import pytest

import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError


class FakeDevice(cirq.Device):
    def __init__(self, qubits):
        self.qubits = qubits

    @property
    def metadata(self):
        return cirq.DeviceMetadata(self.qubits, None)


def _create_device(qubits: Iterable[cirq.GridQubit]):
    return FakeDevice(qubits)


def test_greedy_sequence_search_fails_on_wrong_start_qubit():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    with pytest.raises(ValueError):
        greedy.GreedySequenceSearch(_create_device([q00]), q01)


def test_get_or_search_calls_find_sequence_once():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    search = greedy.GreedySequenceSearch(_create_device([q00, q01]), q00)
    with mock.patch.object(search, '_find_sequence') as find_sequence:
        sequence = [q00, q01]
        find_sequence.return_value = sequence

        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()

        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()


def test_find_sequence_assembles_head_and_tail():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    with mock.patch.object(search, '_sequence_search') as sequence_search:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]
        assert search._find_sequence() == qubits
        sequence_search.assert_has_calls([mock.call(start, []), mock.call(start, tail)])


def test_find_sequence_calls_expand_sequence():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    with mock.patch.object(search, '_sequence_search') as sequence_search, mock.patch.object(
        search, '_expand_sequence'
    ) as expand_sequence:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]

        search._find_sequence()
        expand_sequence.assert_called_once_with(qubits)


def test_search_sequence_calls_choose_next_qubit():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    search = greedy.GreedySequenceSearch(_create_device(qubits), q01)

    with mock.patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [])
        choose_next_qubit.assert_called_once_with(q01, {q01})

    with mock.patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [q00])
        choose_next_qubit.assert_called_once_with(q01, {q00, q01})


def test_search_sequence_assembles_sequence():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    search = greedy.GreedySequenceSearch(_create_device(qubits), q01)

    with mock.patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.side_effect = [q01, q02, None]
        assert search._sequence_search(q00, []) == [q00, q01, q02]


def test_find_path_between_finds_path():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q12 = cirq.GridQubit(1, 2)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    q22 = cirq.GridQubit(2, 2)

    qubits = [q00, q01, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) == [q10, q11]

    # path1: + + +   path2:  +-+-+
    #        |                   |
    #        +   +           +   +
    #        |                   |
    #        + + +           +-+-+
    qubits = [q00, q01, q02, q10, q20, q21, q22, q12]
    path_1 = [q00, q01, q02]
    path_2 = [q00, q10, q20, q21, q22, q12, q02]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) == path_2[1:-1]
    assert search._find_path_between(q02, q00, set(path_1)) == path_2[-2:0:-1]
    assert search._find_path_between(q00, q02, set(path_2)) == path_1[1:-1]
    assert search._find_path_between(q02, q00, set(path_2)) == path_1[-2:0:-1]


def test_find_path_between_does_not_find_path():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q22 = cirq.GridQubit(2, 2)
    q12 = cirq.GridQubit(1, 2)
    qubits = [q00, q01]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None

    qubits = [q00, q01, q10]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None

    # + + +
    # |
    # +
    # |
    # + + +
    qubits = [q00, q01, q02, q10, q20, q22, q12]
    path_1 = [q00, q01, q02]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) is None


def test_expand_sequence_expands_sequence():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q12 = cirq.GridQubit(1, 2)
    q13 = cirq.GridQubit(1, 3)
    q14 = cirq.GridQubit(1, 4)

    # + +  ->  +-+
    # |          |
    # + +      +-+
    qubits = [q00, q01, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01]) == [q00, q10, q11, q01]

    # + +  ->  +-+
    # |          |
    # + +      +-+
    # |        |
    # +        +
    qubits = [q00, q01, q02, q10, q11]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02]) == [q00, q10, q11, q01, q02]

    # +    ->  +
    # |        |
    # + +      +-+
    # |          |
    # + +      +-+
    qubits = [q00, q01, q02, q11, q12]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02]) == [q00, q01, q11, q12, q02]

    # +    ->  +
    # |        |
    # + +      +-+
    # |          |
    # + +      +-+
    # |        |
    # +        +
    qubits = [q00, q01, q02, q03, q11, q12]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03]) == [q00, q01, q11, q12, q02, q03]

    # + +  ->  +-+
    # |          |
    # + +      +-+
    # |        |
    # +        +
    # |        |
    # + +      +-+
    # |          |
    # + +      +-+
    qubits = [q00, q01, q02, q03, q04, q10, q11, q13, q14]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03, q04]) == [
        q00,
        q10,
        q11,
        q01,
        q02,
        q03,
        q13,
        q14,
        q04,
    ]


def test_minimal_sequence_search_chooses_minimal():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q10)
    # + *-+
    #     |
    #     +
    assert search._choose_next_qubit(q10, {q10}) == q20
    assert search._choose_next_qubit(q20, {q10, q20}) == q21


def test_minimal_sequence_search_does_not_use_used():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q10)
    # + *-+
    #
    #     +
    assert search._choose_next_qubit(q10, {q00, q10}) == q20


def test_minimal_sequence_search_returns_none_for_single_node():
    q00 = cirq.GridQubit(0, 0)
    qubits = [q00]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q00)
    assert search._choose_next_qubit(q00, {q00}) is None


def test_minimal_sequence_search_returns_none_when_blocked():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    qubits = [q00, q10]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q10)
    assert search._choose_next_qubit(q10, {q00, q10}) is None


def test_minimal_sequence_search_traverses_grid():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q11 = cirq.GridQubit(1, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q14 = cirq.GridQubit(1, 4)
    q24 = cirq.GridQubit(2, 4)
    q05 = cirq.GridQubit(0, 5)
    qubits = [q00, q01, q11, q02, q03, q04, q05, q14, q24]
    device = _create_device(qubits)
    search = greedy._PickFewestNeighbors(device, q02)
    # (0,0)╌╌(0,1)╌╌START══(0,3)══(0,4)╌╌(0,5)
    #        ╎                    ║
    #        (1,1)                (1,4)
    #                             ║
    #                             (2,4)
    assert search._choose_next_qubit(q02, {q02}) == q03
    assert search._choose_next_qubit(q03, {q02, q03}) == q04
    assert search._choose_next_qubit(q04, {q02, q03, q04}) == q14
    assert search._choose_next_qubit(q14, {q02, q03, q04, q14}) == q24
    assert search._choose_next_qubit(q24, {q02, q03, q04, q14, q24}) is None

    # (0,0)╌╌(0,1)══(0,2)══(0,3)══(0,4)╌╌(0,5)
    #        ║                    ║
    #        (1,1)                (1,4)
    #                             ║
    #                             START
    assert search._choose_next_qubit(q24, {q24}) == q14
    assert search._choose_next_qubit(q14, {q24, q14}) == q04
    assert search._choose_next_qubit(q04, {q24, q14, q04}) == q03
    assert search._choose_next_qubit(q03, {q24, q14, q04, q03}) == q02
    assert search._choose_next_qubit(q02, {q24, q14, q04, q03, q02}) == q01
    assert search._choose_next_qubit(q01, {q24, q14, q04, q03, q02, q01}) in [q00, q11]
    assert search._choose_next_qubit(q00, {q24, q14, q04, q03, q02, q01, q00}) is None
    assert search._choose_next_qubit(q11, {q24, q14, q04, q03, q02, q01, q11}) is None

    # START══(0,1)══(0,2)══(0,3)══(0,4)╌╌(0,5)
    #                             ║
    #                             (1,4)
    #                             ║
    #                             (2,4)
    qubits = [q00, q01, q02, q03, q04, q05, q14, q24]
    device = _create_device(qubits)
    method = greedy.GreedySequenceSearchStrategy('minimal_connectivity')
    assert method.place_line(device, 4) == (q00, q01, q02, q03)
    assert method.place_line(device, 7) == (q00, q01, q02, q03, q04, q14, q24)
    with pytest.raises(NotFoundError):
        _ = method.place_line(device, 8)


def test_largest_sequence_search_chooses_largest():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickLargestArea(_create_device(qubits), q10)
    # + *-+
    #
    #     +
    assert search._choose_next_qubit(q10, {q10}) == q20


def test_largest_sequence_search_does_not_use_used():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickLargestArea(_create_device(qubits), q10)
    # +-* X
    #
    #     +
    assert search._choose_next_qubit(q10, {q10, q20}) == q00


def test_largest_sequence_search_traverses_grid():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q20 = cirq.GridQubit(2, 0)
    q30 = cirq.GridQubit(3, 0)
    q40 = cirq.GridQubit(4, 0)
    q41 = cirq.GridQubit(4, 1)
    q42 = cirq.GridQubit(4, 2)
    q50 = cirq.GridQubit(5, 0)
    qubits = [q00, q10, q11, q20, q30, q40, q50, q41, q42]
    device = _create_device(qubits)
    search = greedy._PickLargestArea(device, q20)
    # + + +-+-+ +
    #         |
    #   +     +
    #         |
    #         +
    assert search._choose_next_qubit(q20, {q20}) == q30
    assert search._choose_next_qubit(q30, {q20, q30}) == q40
    assert search._choose_next_qubit(q40, {q20, q30, q40}) == q41
    assert search._choose_next_qubit(q41, {q20, q30, q40, q41}) == q42
    assert search._choose_next_qubit(q42, {q20, q30, q40, q41, q42}) is None

    method = greedy.GreedySequenceSearchStrategy('largest_area')
    assert method.place_line(device, 7) == GridQubitLineTuple([q00, q10, q20, q30, q40, q41, q42])
    with pytest.raises(NotFoundError):
        _ = method.place_line(device, 8)


def test_largest_collect_unused_collects_all_for_empty():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q12 = cirq.GridQubit(1, 2)
    qubits = [q00, q01, q02, q12]
    start = q01
    search = greedy._PickLargestArea(_create_device(qubits), start)
    assert search._collect_unused(start, set()) == set(qubits)
    assert search._collect_unused(start, {start}) == set(qubits)


def test_largest_collect_unused_collects():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q12 = cirq.GridQubit(1, 2)
    qubits = [q00, q01, q02, q12]
    start = q01
    search = greedy._PickLargestArea(_create_device(qubits), start)
    assert search._collect_unused(start, {q00, q01}) == {q01, q02, q12}


def test_largest_collect_stops_on_used():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q05 = cirq.GridQubit(0, 5)
    q11 = cirq.GridQubit(1, 1)
    q14 = cirq.GridQubit(1, 4)
    q24 = cirq.GridQubit(2, 4)
    qubits = [q00, q01, q11, q02, q03, q04, q05, q14, q24]
    start = q02
    search = greedy._PickLargestArea(_create_device(qubits), start)
    assert search._collect_unused(start, {start, q04}) == {q00, q01, q11, q02, q03}


def test_greedy_search_method_calls_all():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    qubits = [q00, q01]
    length = 2
    method = greedy.GreedySequenceSearchStrategy()
    assert len(method.place_line(_create_device(qubits), length)) == 2


def test_greedy_search_method_fails_when_unknown():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    qubits = [q00, q01]
    length = 2

    method = greedy.GreedySequenceSearchStrategy('fail')
    with pytest.raises(ValueError):
        method.place_line(_create_device(qubits), length)


@mock.patch('cirq_google.line.placement.greedy._PickLargestArea')
@mock.patch('cirq_google.line.placement.greedy._PickFewestNeighbors')
def test_greedy_search_method_calls_largest_only(minimal, largest):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    device = _create_device([q00, q01])
    length = 2
    sequence = [q00, q01]
    largest.return_value.get_or_search.return_value = sequence

    method = greedy.GreedySequenceSearchStrategy('largest_area')
    assert method.place_line(device, length) == GridQubitLineTuple(sequence)

    largest.return_value.get_or_search.assert_called_once_with()
    minimal.return_value.get_or_search.assert_not_called()


@mock.patch('cirq_google.line.placement.greedy._PickLargestArea')
@mock.patch('cirq_google.line.placement.greedy._PickFewestNeighbors')
def test_greedy_search_method_calls_minimal_only(minimal, largest):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    device = _create_device([q00, q01])
    length = 2
    sequence = [q00, q01]
    minimal.return_value.get_or_search.return_value = sequence

    method = greedy.GreedySequenceSearchStrategy('minimal_connectivity')
    assert method.place_line(device, length) == GridQubitLineTuple(sequence)

    largest.return_value.get_or_search.assert_not_called()
    minimal.return_value.get_or_search.assert_called_once_with()


def test_greedy_search_method_returns_longest():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    device = _create_device([q00, q10])
    length = 1

    method = greedy.GreedySequenceSearchStrategy()
    assert method.place_line(device, length) == GridQubitLineTuple([q00])


def test_greedy_search_method_returns_empty_when_empty():
    device = _create_device([])
    length = 0
    method = greedy.GreedySequenceSearchStrategy()
    assert method.place_line(device, length) == GridQubitLineTuple()
