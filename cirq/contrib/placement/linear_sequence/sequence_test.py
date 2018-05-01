"""Tests for sequence helpers.
"""

from cirq.contrib.placement.linear_sequence.sequence import \
    longest_sequence_index
from cirq.google import XmonQubit


def test_single_sequence():
    assert longest_sequence_index([[XmonQubit(0, 0)]]) == 0


def test_longest_sequence():
    q00, q01, q02, q03 = [XmonQubit(0, x) for x in range(4)]
    assert longest_sequence_index([[q00], [q01, q02, q03]]) == 1


def test_multiple_longest_sequences():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    q10 = XmonQubit(1, 0)
    q20 = XmonQubit(2, 0)
    assert longest_sequence_index([[q00], [q01, q02], [q10, q20]]) == 1


def test_empty_list():
    assert longest_sequence_index([]) is None
