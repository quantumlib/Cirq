# Copyright 2018 Google LLC
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

import numpy as np

import cirq
from cirq import ops
from cirq.extension import Extensions
from cirq.google import MergeRotations


def assert_optimizes(before, after, optimizer=None):
    if optimizer is None:
        optimizer = MergeRotations()
    optimizer.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        cirq.DropNegligible(),
        cirq.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        print("before:", before)
        print("after:", after)
    assert before == after


def test_leaves_singleton():
    m = MergeRotations(cirq.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitId()
    c = cirq.Circuit([cirq.Moment([ops.X(q)])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([cirq.Moment([ops.X(q)])])


def test_combines_sequence():
    m = MergeRotations(cirq.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitId()
    c = cirq.Circuit([
        cirq.Moment([ops.X(q)**0.5]),
        cirq.Moment([ops.Z(q)**0.5]),
        cirq.Moment([ops.X(q)**-0.5]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([
        cirq.Moment([ops.Y(q)**0.5]),
        cirq.Moment(),
        cirq.Moment(),
    ])


def test_removes_identity_sequence():
    q = ops.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([ops.Z(q)]),
            cirq.Moment([ops.H(q)]),
            cirq.Moment([ops.X(q)]),
            cirq.Moment([ops.H(q)]),
        ]),
        after = cirq.Circuit())


def test_stopped_at_2qubit():
    m = MergeRotations(cirq.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = cirq.Circuit([
        cirq.Moment([ops.Z(q)]),
        cirq.Moment([ops.H(q)]),
        cirq.Moment([ops.X(q)]),
        cirq.Moment([ops.H(q)]),
        cirq.Moment([ops.CZ(q, q2)]),
        cirq.Moment([ops.H(q)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment(),
        cirq.Moment([ops.CZ(q, q2)]),
        cirq.Moment([ops.H(q)]),
    ])


def test_ignores_2qubit_target():
    m = MergeRotations(cirq.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = cirq.Circuit([
        cirq.Moment([ops.CZ(q, q2)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([cirq.Moment([ops.CZ(q, q2)])])


def test_extension():
    class DummyGate(ops.Gate):
        pass

    optimizer = MergeRotations(extensions=Extensions({
        ops.KnownMatrixGate: {
            DummyGate: lambda _: ops.SingleQubitMatrixGate(
                np.array([[0, 1], [1, 0]]))
        }
    }))

    q = ops.QubitId()
    c = cirq.Circuit([
        cirq.Moment([DummyGate().on(q)]),
    ])
    assert_optimizes(
        before=c,
        after=cirq.Circuit([cirq.Moment([ops.X(q)])]),
        optimizer=optimizer)
