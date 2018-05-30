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

from cirq.contrib.placement.linear_sequence.sequence import (
    longest_sequence_index
)
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
