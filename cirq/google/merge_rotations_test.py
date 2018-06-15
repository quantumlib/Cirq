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

import numpy as np

from cirq import circuits
from cirq import ops
from cirq.extension import Extensions
from cirq.google import MergeRotations


def assert_optimizes(before, after, optimizer=None):
    if optimizer is None:
        optimizer = MergeRotations()
    optimizer.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        circuits.DropNegligible(),
        circuits.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        # coverage: ignore
        print("before:", before)
        print("after:", after)
    assert before == after


def test_leaves_singleton():
    m = MergeRotations(0.000001)
    q = ops.QubitId()
    c = circuits.Circuit([circuits.Moment([ops.X(q)])])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment([ops.X(q)])])


def test_combines_sequence():
    m = MergeRotations(0.000001)
    q = ops.QubitId()
    c = circuits.Circuit([
        circuits.Moment([ops.X(q)**0.5]),
        circuits.Moment([ops.Z(q)**0.5]),
        circuits.Moment([ops.X(q)**-0.5]),
    ])

    assert (m.optimization_at(c, 0, c.operation_at(q, 0)) ==
            circuits.PointOptimizationSummary(clear_span=3,
                                              clear_qubits=[q],
                                              new_operations=ops.Y(q)**0.5))


def test_removes_identity_sequence():
    q = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.Z(q)]),
            circuits.Moment([ops.H(q)]),
            circuits.Moment([ops.X(q)]),
            circuits.Moment([ops.H(q)]),
        ]),
        after = circuits.Circuit())


def test_stopped_at_2qubit():
    m = MergeRotations(0.000001)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = circuits.Circuit([
        circuits.Moment([ops.Z(q)]),
        circuits.Moment([ops.H(q)]),
        circuits.Moment([ops.X(q)]),
        circuits.Moment([ops.H(q)]),
        circuits.Moment([ops.CZ(q, q2)]),
        circuits.Moment([ops.H(q)]),
    ])

    assert (m.optimization_at(c, 0, c.operation_at(q, 0)) ==
            circuits.PointOptimizationSummary(clear_span=4,
                                              clear_qubits=[q],
                                              new_operations=[]))


def test_ignores_2qubit_target():
    m = MergeRotations(0.000001)
    q = ops.QubitId()
    q2 = ops.QubitId()
    c = circuits.Circuit([
        circuits.Moment([ops.CZ(q, q2)]),
    ])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    assert c == circuits.Circuit([circuits.Moment([ops.CZ(q, q2)])])


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
    c = circuits.Circuit([
        circuits.Moment([DummyGate().on(q)]),
    ])
    assert_optimizes(
        before=c,
        after=circuits.Circuit([circuits.Moment([ops.X(q)])]),
        optimizer=optimizer)
