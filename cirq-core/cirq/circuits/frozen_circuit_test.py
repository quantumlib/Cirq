# Copyright 2020 The Cirq Developers
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
"""Tests exclusively for FrozenCircuits.

Behavior shared with Circuit is tested with parameters in circuit_test.py.
"""

import pytest

import cirq


def test_from_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    assert cirq.FrozenCircuit.from_moments(
        [cirq.X(a), cirq.Y(b)],
        [cirq.X(c)],
        [],
        cirq.Z(d),
        [cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')],
    ) == cirq.FrozenCircuit(
        cirq.Moment(cirq.X(a), cirq.Y(b)),
        cirq.Moment(cirq.X(c)),
        cirq.Moment(),
        cirq.Moment(cirq.Z(d)),
        cirq.Moment(cirq.measure(a, b, key='ab'), cirq.measure(c, d, key='cd')),
    )


def test_freeze_and_unfreeze():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(b))

    f = c.freeze()
    # Circuits equal their frozen versions, similar to set(x) == frozenset(x).
    assert f == c
    assert cirq.approx_eq(f, c)

    # Freezing a FrozenCircuit will return the original.
    ff = f.freeze()
    assert ff is f

    unf = f.unfreeze()
    assert unf.moments == c.moments
    assert unf is not c

    # Unfreezing always returns a copy.
    cc = c.unfreeze()
    assert cc is not c

    fcc = cc.freeze()
    assert fcc.moments == f.moments
    assert fcc is not f


def test_immutable():
    q = cirq.LineQubit(0)
    c = cirq.FrozenCircuit(cirq.X(q), cirq.H(q))

    with pytest.raises(AttributeError, match="can't set attribute"):
        c.moments = (cirq.Moment(cirq.H(q)), cirq.Moment(cirq.X(q)))
