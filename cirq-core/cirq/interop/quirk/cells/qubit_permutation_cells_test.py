# Copyright 2019 The Cirq Developers
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
from cirq.interop.quirk.cells.qubit_permutation_cells import QuirkQubitPermutationGate
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_equality():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: QuirkQubitPermutationGate('a', 'b', [0, 1]))
    eq.add_equality_group(QuirkQubitPermutationGate('x', 'b', [0, 1]))
    eq.add_equality_group(QuirkQubitPermutationGate('a', 'X', [0, 1]))
    eq.add_equality_group(QuirkQubitPermutationGate('a', 'b', [1, 0]))


def test_repr():
    cirq.testing.assert_equivalent_repr(QuirkQubitPermutationGate('a', 'b', [0, 1]))


def test_right_rotate():
    assert_url_to_circuit_returns(
        '{"cols":[["X",">>4",1,1,1,"X"]]}',
        diagram="""
0: ───X───────────────────

1: ───right_rotate[0>3]───
      │
2: ───right_rotate[1>0]───
      │
3: ───right_rotate[2>1]───
      │
4: ───right_rotate[3>2]───

5: ───X───────────────────
        """,
        maps={
            0b_000000: 0b_100001,
            0b_000010: 0b_100101,
            0b_000100: 0b_101001,
            0b_001000: 0b_110001,
            0b_010000: 0b_100011,
            0b_011110: 0b_111111,
            0b_010100: 0b_101011,
        },
    )


def test_left_rotate():
    assert_url_to_circuit_returns(
        '{"cols":[["<<4"]]}',
        maps={
            0b_0000: 0b_0000,
            0b_0001: 0b_1000,
            0b_0010: 0b_0001,
            0b_0100: 0b_0010,
            0b_1000: 0b_0100,
            0b_1111: 0b_1111,
            0b_1010: 0b_0101,
        },
    )


def test_reverse():
    assert_url_to_circuit_returns(
        '{"cols":[["rev4"]]}',
        maps={
            0b_0000: 0b_0000,
            0b_0001: 0b_1000,
            0b_0010: 0b_0100,
            0b_0100: 0b_0010,
            0b_1000: 0b_0001,
            0b_1111: 0b_1111,
            0b_1010: 0b_0101,
        },
    )
    assert_url_to_circuit_returns(
        '{"cols":[["rev3"]]}',
        maps={
            0b_000: 0b_000,
            0b_001: 0b_100,
            0b_010: 0b_010,
            0b_100: 0b_001,
            0b_111: 0b_111,
            0b_101: 0b_101,
        },
    )


def test_interleave():
    assert_url_to_circuit_returns(
        '{"cols":[["weave5"]]}',
        maps={
            0b_00000: 0b_00000,
            0b_00001: 0b_00010,
            0b_00010: 0b_01000,
            0b_00100: 0b_00001,
            0b_01000: 0b_00100,
            0b_10000: 0b_10000,
            0b_00011: 0b_01010,
            0b_11111: 0b_11111,
        },
    )
    assert_url_to_circuit_returns(
        '{"cols":[["weave6"]]}',
        maps={
            0b_000000: 0b_000000,
            0b_000001: 0b_000001,
            0b_000010: 0b_000100,
            0b_000100: 0b_010000,
            0b_001000: 0b_000010,
            0b_010000: 0b_001000,
            0b_100000: 0b_100000,
            0b_000111: 0b_010101,
            0b_111111: 0b_111111,
        },
    )


def test_deinterleave():
    assert_url_to_circuit_returns(
        '{"cols":[["split5"]]}',
        maps={
            0b_00000: 0b_00000,
            0b_00001: 0b_00100,
            0b_00010: 0b_00001,
            0b_00100: 0b_01000,
            0b_01000: 0b_00010,
            0b_10000: 0b_10000,
            0b_01010: 0b_00011,
            0b_11111: 0b_11111,
        },
    )
    assert_url_to_circuit_returns(
        '{"cols":[["split6"]]}',
        maps={
            0b_000000: 0b_000000,
            0b_000001: 0b_000001,
            0b_000010: 0b_001000,
            0b_000100: 0b_000010,
            0b_001000: 0b_010000,
            0b_010000: 0b_000100,
            0b_100000: 0b_100000,
            0b_010101: 0b_000111,
            0b_111111: 0b_111111,
        },
    )
