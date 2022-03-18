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

import cirq


def assert_optimizes(before, after, measure_only_moment=True):
    with cirq.testing.assert_deprecated(
        "Use cirq.synchronize_terminal_measurements", deadline='v1.0'
    ):
        opt = cirq.SynchronizeTerminalMeasurements(measure_only_moment)
        opt(before)
        assert before == after


def test_no_move():
    q1 = cirq.NamedQubit('q1')
    before = cirq.Circuit([cirq.Moment([cirq.H(q1)])])
    after = before
    assert_optimizes(before=before, after=after)


def test_simple_align():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.measure(q2)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.Z(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.measure(q2)]),
        ]
    )
    assert_optimizes(before=before, after=after)


def test_simple_partial_align():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.Z(q1), cirq.measure(q2)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.Z(q1)]),
            cirq.Moment([cirq.measure(q2)]),
        ]
    )
    assert_optimizes(before=before, after=after)


def test_slide_forward_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.measure(q2), cirq.measure(q3)]),
        ]
    )
    after = cirq.Circuit(
        [cirq.Moment([cirq.H(q1)]), cirq.Moment([cirq.measure(q2), cirq.measure(q3)])]
    )
    assert_optimizes(before=before, after=after)


def test_no_slide_forward_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.measure(q2), cirq.measure(q3)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.measure(q2), cirq.measure(q3)]),
        ]
    )
    assert_optimizes(before=before, after=after, measure_only_moment=False)


def test_blocked_shift_one():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1), cirq.measure(q2)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1)]),
            cirq.Moment([cirq.measure(q2)]),
        ]
    )
    assert_optimizes(before=before, after=after)


def test_complex_move():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1), cirq.measure(q2)]),
            cirq.Moment([cirq.H(q3)]),
            cirq.Moment([cirq.X(q1), cirq.measure(q3)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1)]),
            cirq.Moment([cirq.H(q3)]),
            cirq.Moment([cirq.X(q1)]),
            cirq.Moment([cirq.measure(q2), cirq.measure(q3)]),
        ]
    )
    assert_optimizes(before=before, after=after)


def test_complex_move_no_slide():
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    before = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1), cirq.measure(q2)]),
            cirq.Moment([cirq.H(q3)]),
            cirq.Moment([cirq.X(q1), cirq.measure(q3)]),
        ]
    )
    after = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q1), cirq.H(q2)]),
            cirq.Moment([cirq.measure(q1), cirq.Z(q2)]),
            cirq.Moment([cirq.H(q1)]),
            cirq.Moment([cirq.H(q3)]),
            cirq.Moment([cirq.X(q1), cirq.measure(q2), cirq.measure(q3)]),
        ]
    )
    assert_optimizes(before=before, after=after, measure_only_moment=False)
