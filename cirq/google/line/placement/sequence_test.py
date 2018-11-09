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

import cirq
from cirq.google.line.placement.sequence import (
    GridQubitLineTuple,
    NotFoundError
)


def test_best_of_gets_longest_needs_minimum():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)

    assert GridQubitLineTuple.best_of([[]], 0) == ()
    assert GridQubitLineTuple.best_of([[], [q00]], 0) == ()
    assert GridQubitLineTuple.best_of([[q00], []], 0) == ()
    assert GridQubitLineTuple.best_of([[], [q00]], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00], []], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00, q01], [q00]], 1) == (q00,)
    assert GridQubitLineTuple.best_of([[q00, q01], [q00]], 2) == (q00, q01)
    assert GridQubitLineTuple.best_of([[q00, q01]], 2) == (q00, q01)

    assert GridQubitLineTuple.best_of([], 0) == ()
    with pytest.raises(NotFoundError):
        _ = GridQubitLineTuple.best_of([[]], 1)
    with pytest.raises(NotFoundError):
        _ = GridQubitLineTuple.best_of([[q00]], 2)


def test_line_placement_str():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    placement = GridQubitLineTuple([q00, q01, q02])
    assert str(placement).strip() == """
(0, 0)━━(0, 1)━━(0, 2)
    """.strip()


def test_line_placement_to_str():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    placement = GridQubitLineTuple([q02, q01, q00, q10, q11])
    assert str(placement).strip() == """
(0, 0)━━(0, 1)━━(0, 2)
┃
(1, 0)━━(1, 1)
    """.strip()
