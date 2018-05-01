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
from unittest.mock import call, patch

from cirq.contrib.placement.linear_sequence.greedy import GreedySequenceSearch, \
    MinimalConnectivityGreedySequenceSearch, LargestAreaGreedySequenceSearch, \
    greedy_sequence
from cirq.google import XmonDevice, XmonQubit
from cirq.value import Duration


def _create_device(qubits: Iterable[XmonQubit]):
    return XmonDevice(Duration(nanos=0), Duration(nanos=0), Duration(nanos=0),
                      qubits)


def test_get_or_search_calls_find_sequence_once():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    search = GreedySequenceSearch(_create_device([q00, q01]), q00)
    with patch.object(search, '_find_sequence') as find_sequence:
        sequence = [q00, q01]
        find_sequence.return_value = sequence

        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()

        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()


def test_find_sequence_assembles_head_and_tail():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = GreedySequenceSearch(_create_device(qubits), start)
    with patch.object(search, '_sequence_search') as sequence_search:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]
        assert search._find_sequence() == qubits
        sequence_search.assert_has_calls(
            [call(start, []), call(start, tail)])


def test_find_sequence_calls_expand_sequence():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    qubits = [q00, q01, q02]
    start = q01
    search = GreedySequenceSearch(_create_device(qubits), start)
    with patch.object(
            search, '_sequence_search') as sequence_search, patch.object(
        search, '_expand_sequence') as expand_sequence:
        head = [q01, q00]
        tail = [q01, q02]
        sequence_search.side_effect = [tail, head]

        search._find_sequence()
        expand_sequence.assert_called_once_with(qubits)


def test_search_sequence_calls_choose_next_qubit():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    qubits = [q00, q01, q02]
    search = GreedySequenceSearch(_create_device(qubits), q01)

    with patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [])
        choose_next_qubit.assert_called_once_with(q01, {q01})

    with patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [q00])
        choose_next_qubit.assert_called_once_with(q01, {q00, q01})


def test_search_sequence_assembles_sequence():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    qubits = [q00, q01, q02]
    search = GreedySequenceSearch(_create_device(qubits), q01)

    with patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.side_effect = [q01, q02, None]
        assert search._sequence_search(q00, []) == [q00, q01, q02]


def test_find_path_between_finds_path():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q10 = XmonQubit(1, 0)
    q11 = XmonQubit(1, 1)
    q12 = XmonQubit(1, 2)
    q20 = XmonQubit(2, 0)
    q21 = XmonQubit(2, 1)
    q22 = XmonQubit(2, 2)

    qubits = [q00, q01, q10, q11]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
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
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) == path_2[1:-1]
    assert search._find_path_between(q02, q00, set(path_1)) == path_2[-2:0:-1]
    assert search._find_path_between(q00, q02, set(path_2)) == path_1[1:-1]
    assert search._find_path_between(q02, q00, set(path_2)) == path_1[-2:0:-1]


def test_find_path_between_does_not_find_path():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    q22 = XmonQubit(2, 2)
    q12 = XmonQubit(1, 2)
    qubits = [q00, q01]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None

    qubits = [q00, q01, q10]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None

    # + + +
    # |
    # +
    # |
    # + + +
    qubits = [q00, q01, q02, q10, q20, q22, q12]
    path_1 = [q00, q01, q02]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) is None


def test_expand_sequence_expands_sequence():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q03 = XmonQubit(0, 3)
    q04 = XmonQubit(0, 4)
    q10 = XmonQubit(1, 0)
    q11 = XmonQubit(1, 1)
    q12 = XmonQubit(1, 2)
    q13 = XmonQubit(1, 3)
    q14 = XmonQubit(1, 4)

    # + +  ->  +-+
    # |          |
    # + +      +-+
    qubits = [q00, q01, q10, q11]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01]) == [q00, q10, q11, q01]

    # + +  ->  +-+
    # |          |
    # + +      +-+
    # |        |
    # +        +
    qubits = [q00, q01, q02, q10, q11]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02]) == [q00, q10, q11, q01, q02]

    # +    ->  +
    # |        |
    # + +      +-+
    # |          |
    # + +      +-+
    qubits = [q00, q01, q02, q11, q12]
    start = q00
    search = GreedySequenceSearch(_create_device(qubits), start)
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
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03]) == [q00, q01, q11, q12,
                                                             q02, q03]

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
    search = GreedySequenceSearch(_create_device(qubits), start)
    assert search._expand_sequence([q00, q01, q02, q03, q04]) == [q00, q10, q11,
                                                                  q01, q02, q03,
                                                                  q13, q14, q04]


def test_sequence_search_chooses_minimal():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    q21 = XmonQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = MinimalConnectivityGreedySequenceSearch(_create_device(qubits),
                                                     q10)
    # +-* +
    #
    #     +
    assert search._choose_next_qubit(q10, {q10}) == q00


def test_sequence_search_does_not_use_used():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    q21 = XmonQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = MinimalConnectivityGreedySequenceSearch(_create_device(qubits),
                                                     q10)
    # + *-+
    #
    #     +
    assert search._choose_next_qubit(q10, {q00, q10}) == q20


def test_sequence_search_returns_none_for_single_node():
    q00 = XmonQubit(0, 0)
    qubits = [q00]
    search = MinimalConnectivityGreedySequenceSearch(_create_device(qubits),
                                                     q00)
    assert search._choose_next_qubit(q00, {q00}) is None


def test_sequence_search_returns_none_when_blocked():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    qubits = [q00, q10]
    search = MinimalConnectivityGreedySequenceSearch(_create_device(qubits),
                                                     q10)
    assert search._choose_next_qubit(q10, {q00, q10}) is None


def test_sequence_search_traverses_grid():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q11 = XmonQubit(1, 1)
    q20 = XmonQubit(2, 0)
    q30 = XmonQubit(3, 0)
    q40 = XmonQubit(4, 0)
    q41 = XmonQubit(4, 1)
    q42 = XmonQubit(4, 2)
    q50 = XmonQubit(5, 0)
    qubits = [q00, q10, q11, q20, q30, q40, q50, q41, q42]
    search = MinimalConnectivityGreedySequenceSearch(_create_device(qubits),
                                                     q20)
    # + + *-+-+-+
    #
    #   +     +
    #
    #         +
    assert search._choose_next_qubit(q20, {q20}) == q30
    assert search._choose_next_qubit(q30, {q20, q30}) == q40
    assert search._choose_next_qubit(q40, {q20, q30, q40}) == q50
    assert search._choose_next_qubit(q50, {q20, q30, q40, q50}) is None


def test_sequence_search_chooses_largest():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    q21 = XmonQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), q10)
    # + *-+
    #
    #     +
    assert search._choose_next_qubit(q10, {q10}) == q20


def test_sequence_search_does_not_use_used():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    q21 = XmonQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), q10)
    # +-* X
    #
    #     +
    assert search._choose_next_qubit(q10, {q10, q20}) == q00


def test_sequence_search_traverses_grid():
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    q11 = XmonQubit(1, 1)
    q20 = XmonQubit(2, 0)
    q30 = XmonQubit(3, 0)
    q40 = XmonQubit(4, 0)
    q41 = XmonQubit(4, 1)
    q42 = XmonQubit(4, 2)
    q50 = XmonQubit(5, 0)
    qubits = [q00, q10, q11, q20, q30, q40, q50, q41, q42]
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), q20)
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


def test_collect_unused_collects_all_for_empty():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q12 = XmonQubit(1, 2)
    qubits = [q00, q01, q02, q12]
    start = q01
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), start)
    assert search._collect_unused(start, set()) == set(qubits)
    assert search._collect_unused(start, {start}) == set(qubits)


def test_collect_unused_collects():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q12 = XmonQubit(1, 2)
    qubits = [q00, q01, q02, q12]
    start = q01
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), start)
    assert search._collect_unused(start, {q00, q01}) == {q01, q02, q12}


def test_collect_stops_on_used():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q03 = XmonQubit(0, 3)
    q04 = XmonQubit(0, 4)
    q05 = XmonQubit(0, 5)
    q11 = XmonQubit(1, 1)
    q14 = XmonQubit(1, 4)
    q24 = XmonQubit(2, 4)
    qubits = [q00, q01, q11, q02, q03, q04, q05, q14, q24]
    start = q02
    search = LargestAreaGreedySequenceSearch(_create_device(qubits), start)
    assert search._collect_unused(start, {start, q04}) == {q00, q01, q11, q02,
                                                           q03}


@patch(
    'cirq.contrib.placement.linear_sequence.greedy.LargestAreaGreedySequenceSearch')
@patch(
    'cirq.contrib.placement.linear_sequence.greedy.MinimalConnectivityGreedySequenceSearch')
def test_greedy_sequence_calls_all(largest, minimal):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    qubits = [q00, q01]
    largest_instance = largest.return_value
    minimal_instance = minimal.return_value
    greedy_sequence(_create_device(qubits))
    largest.assert_called_once_with(_create_device(qubits), q00)
    largest_instance.get_or_search.assert_called_once_with()
    minimal.assert_called_once_with(_create_device(qubits), q00)
    minimal_instance.get_or_search.assert_called_once_with()


@patch(
    'cirq.contrib.placement.linear_sequence.greedy.LargestAreaGreedySequenceSearch')
@patch(
    'cirq.contrib.placement.linear_sequence.greedy.MinimalConnectivityGreedySequenceSearch')
def test_greedy_sequence_returns_longest(largest, minimal):
    q00 = XmonQubit(0, 0)
    q10 = XmonQubit(1, 0)
    sequence_short = [q00]
    sequence_long = [q00, q10]
    largest.return_value.get_or_search.return_value = sequence_short
    minimal.return_value.get_or_search.return_value = sequence_long
    assert greedy_sequence(_create_device([])) == [sequence_long]


@patch(
    'cirq.contrib.placement.linear_sequence.greedy.LargestAreaGreedySequenceSearch')
@patch(
    'cirq.contrib.placement.linear_sequence.greedy.MinimalConnectivityGreedySequenceSearch')
def test_greedy_sequence_returns_empty_when_empty(largest, minimal):
    largest.return_value.get_or_search.return_value = []
    minimal.return_value.get_or_search.return_value = []
    assert greedy_sequence(_create_device([])) == []
