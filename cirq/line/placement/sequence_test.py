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

import pytest

from cirq.line.placement.sequence import (
    LinePlacement,
    LineSequence,
    NotFoundError
)
from cirq.devices import GridQubit
from cirq.google import XmonDevice
from cirq.testing import EqualsTester
from cirq.testing.mock import mock
from cirq.value import Duration


def _create_device(qubits: Iterable[GridQubit]):
    return XmonDevice(Duration(nanos=0), Duration(nanos=0), Duration(nanos=0),
                      qubits)


def test_line_sequence_eq():
    eq = EqualsTester()
    eq.make_equality_group(lambda: LineSequence([]))
    eq.make_equality_group(lambda: LineSequence([GridQubit(0, 0)]))
    eq.make_equality_group(
        lambda: LineSequence([GridQubit(1, 0), GridQubit(0, 0)]))


def test_line_placement_eq():
    eq = EqualsTester()
    eq.make_equality_group(lambda: LinePlacement(_create_device([]), 0, []))
    eq.make_equality_group(
        lambda: LinePlacement(_create_device([GridQubit(0, 0)]), 1,
                              [LineSequence([GridQubit(0, 0)])]))
    eq.make_equality_group(
        lambda: LinePlacement(
            _create_device([GridQubit(0, 0), GridQubit(0, 1)]), 2,
            [LineSequence([GridQubit(0, 0)]),
             LineSequence([GridQubit(0, 1)])]))


def test_line_placement_get_calls_longest():
    seq = LineSequence([])
    placement = LinePlacement(_create_device([]), 0, [seq])
    with mock.patch.object(placement, 'longest') as longest:
        longest.return_value = seq
        assert placement.get() == seq
        longest.assert_called_once_with()


def test_line_placement_get_raises_for_none():
    placement = LinePlacement(_create_device([]), 1, [])
    with mock.patch.object(placement, 'longest') as longest:
        longest.return_value = None
        with pytest.raises(NotFoundError):
            placement.get()


def test_line_placement_get_raises_for_too_short():
    seq = LineSequence([])
    placement = LinePlacement(_create_device([]), 1, [seq])
    with mock.patch.object(placement, 'longest') as longest:
        longest.return_value = seq
        with pytest.raises(NotFoundError):
            placement.get()


def test_line_placement_longest_empty_sequence():
    seq = LineSequence([])
    assert LinePlacement(_create_device([]), 0, [seq]).longest() == seq


def test_line_placement_longest_single_sequence():
    seq = LineSequence([GridQubit(0, 0)])
    assert LinePlacement(_create_device([]), 0, [seq]).longest() == seq


def test_line_placement_longest_longest_sequence():
    q00, q01, q02, q03 = [GridQubit(0, x) for x in range(4)]
    device = _create_device([q00, q01, q02, q03])
    seq1 = LineSequence([q00])
    seq2 = LineSequence([q01, q02, q03])
    assert LinePlacement(device, 0, [seq1, seq2]).longest() == seq2


def test_line_placement_longest_multiple_longest_sequences():
    q00 = GridQubit(0, 0)
    q01 = GridQubit(0, 1)
    q02 = GridQubit(0, 2)
    q10 = GridQubit(1, 0)
    q20 = GridQubit(2, 0)
    device = _create_device([q00, q01, q02, q10, q20])
    seq1 = LineSequence([q00])
    seq2 = LineSequence([q01, q02])
    seq3 = LineSequence([q10, q20])
    assert LinePlacement(device, 0, [seq1, seq2, seq3]).longest() == seq2


def test_line_placement_longest_empty_list():
    assert LinePlacement(_create_device([]), 0, []).longest() is None


def test_line_placement_str():
    q00 = GridQubit(0, 0)
    q01 = GridQubit(0, 1)
    q02 = GridQubit(0, 2)
    q10 = GridQubit(1, 0)
    q11 = GridQubit(1, 1)
    device = _create_device([q00, q01, q02, q10, q11])
    seq1 = LineSequence([q00, q01, q02])
    seq2 = LineSequence([q10, q11])
    placement = LinePlacement(device, 0, [seq1, seq2])
    assert str(placement).strip() == """
(0, 0)━━━(0, 1)━━━(0, 2)


(1, 0)━━━(1, 1)
    """.strip()


def test_line_placement_to_str():
    q00 = GridQubit(0, 0)
    q01 = GridQubit(0, 1)
    q02 = GridQubit(0, 2)
    q10 = GridQubit(1, 0)
    q11 = GridQubit(1, 1)
    device = _create_device([q00, q01, q02, q10, q11])
    seq1 = LineSequence([q00, q01, q02])
    seq2 = LineSequence([q10, q11])
    placement = LinePlacement(device, 0, [seq1, seq2])
    assert placement._to_str(True).strip() == """
(0, 0)━━━(0, 1)━━━(0, 2)
│        │
│        │
(1, 0)━━━(1, 1)
    """.strip()