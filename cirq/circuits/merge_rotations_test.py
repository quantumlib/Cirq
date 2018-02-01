# Copyright 2017 Google LLC
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

from cirq import circuits
from cirq import ops


def test_leaves_singleton():
    m = circuits.MergeRotations(circuits.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitLoc(0, 0)
    c = circuits.Circuit([circuits.Moment([ops.X(q)])])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment([ops.X(q)])])


def test_combines_sequence():
    m = circuits.MergeRotations(circuits.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitLoc(0, 0)
    c = circuits.Circuit([
        circuits.Moment([(ops.X**0.5)(q)]),
        circuits.Moment([(ops.Z**0.5)(q)]),
        circuits.Moment([(ops.X**-0.5)(q)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([
        circuits.Moment([(ops.Y**0.5)(q)]),
        circuits.Moment(),
        circuits.Moment(),
    ])


def test_removes_identity_sequence():
    m = circuits.MergeRotations(circuits.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitLoc(0, 0)
    c = circuits.Circuit([
        circuits.Moment([ops.Z(q)]),
        circuits.Moment([ops.H(q)]),
        circuits.Moment([ops.X(q)]),
        circuits.Moment([ops.H(q)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([
        circuits.Moment(),
        circuits.Moment(),
        circuits.Moment(),
        circuits.Moment(),
    ])


def test_stopped_at_2qubit():
    m = circuits.MergeRotations(circuits.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitLoc(0, 0)
    q2 = ops.QubitLoc(0, 1)
    c = circuits.Circuit([
        circuits.Moment([ops.Z(q)]),
        circuits.Moment([ops.H(q)]),
        circuits.Moment([ops.X(q)]),
        circuits.Moment([ops.H(q)]),
        circuits.Moment([ops.CZ(q, q2)]),
        circuits.Moment([ops.H(q)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([
        circuits.Moment(),
        circuits.Moment(),
        circuits.Moment(),
        circuits.Moment(),
        circuits.Moment([ops.CZ(q, q2)]),
        circuits.Moment([ops.H(q)]),
    ])


def test_ignores_2qubit_target():
    m = circuits.MergeRotations(circuits.InsertStrategy.INLINE, 0.000001)
    q = ops.QubitLoc(0, 0)
    q2 = ops.QubitLoc(0, 1)
    c = circuits.Circuit([
        circuits.Moment([ops.CZ(q, q2)]),
    ])

    m.optimize_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment([ops.CZ(q, q2)])])
