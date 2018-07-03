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

from cirq.line.placement.sequence import (
    LinePlacement,
    LineSequence,
    NotFoundError
)
from cirq.google import XmonQubit
from cirq.testing import EqualsTester
from cirq.testing.mock import mock


def test_line_sequence_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: LineSequence([]))
    eq.make_equality_pair(lambda: LineSequence([XmonQubit(0, 0)]))
    eq.make_equality_pair(
        lambda: LineSequence([XmonQubit(1, 0), XmonQubit(0, 0)]))


def test_line_placement_eq():
    eq = EqualsTester()
    eq.make_equality_pair(lambda: LinePlacement(0, []))
    eq.make_equality_pair(
        lambda: LinePlacement(1, [LineSequence([XmonQubit(0, 0)])]))
    eq.make_equality_pair(
        lambda: LinePlacement(2, [LineSequence([XmonQubit(0, 0)]),
                                  LineSequence([XmonQubit(0, 1)])]))


def test_line_placement_get_calls_longest():
    seq = LineSequence([])
    placement = LinePlacement(0, [seq])
    with mock.patch.object(placement, 'longest') as longest:
        longest.return_value = seq
        assert placement.get() == seq
        longest.assert_called_once_with()


def test_line_placement_get_raises_for_too_short():
    seq = LineSequence([])
    placement = LinePlacement(1, [seq])
    with mock.patch.object(placement, 'longest') as longest:
        longest.return_value = seq
        with pytest.raises(NotFoundError):
            placement.get()


def test_line_placement_longest_empty_sequence():
    seq = LineSequence([])
    assert LinePlacement(0, [seq]).longest() == seq


def test_line_placement_longest_single_sequence():
    seq = LineSequence([XmonQubit(0, 0)])
    assert LinePlacement(0, [seq]).longest() == seq


def test_line_placement_longest_longest_sequence():
    q00, q01, q02, q03 = [XmonQubit(0, x) for x in range(4)]
    seq1 = LineSequence([q00])
    seq2 = LineSequence([q01, q02, q03])
    assert LinePlacement(0, [seq1, seq2]).longest() == seq2


def test_line_placement_longest_multiple_longest_sequences():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    seq1 = LineSequence([q00])
    seq2 = LineSequence([q01, q02])
    seq3 = LineSequence([q10, q20])
    assert LinePlacement(0, [seq1, seq2, seq3]).longest() == seq2


def test_line_placement_longest_empty_list():
    assert LinePlacement(0, []).longest() is None
