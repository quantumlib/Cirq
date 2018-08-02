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

import cirq


def assert_optimizes(before: cirq.Circuit,
                     expected: cirq.Circuit,
                     optimizer: cirq.OptimizationPass=None):
    if optimizer is None:
        optimizer = cirq.MergeSingleQubitGates()
    optimizer.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        cirq.DropNegligible(),
        cirq.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(expected)

    try:
        assert before == expected
    except AssertionError:  # coverage: ignore
        # coverage: ignore
        print("BEFORE")
        print(before)
        print("EXPECTED")
        print(expected)
        raise


def test_leaves_singleton():
    m = cirq.MergeSingleQubitGates()
    q = cirq.QubitId()
    c = cirq.Circuit([cirq.Moment([cirq.X(q)])])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([cirq.Moment([cirq.X(q)])])


def test_combines_sequence():
    m = cirq.MergeSingleQubitGates()
    q = cirq.QubitId()
    c = cirq.Circuit.from_ops(
        cirq.X(q)**0.5,
        cirq.Z(q)**0.5,
        cirq.X(q)**-0.5)

    opt_summary = m.optimization_at(c, 0, c.operation_at(q, 0))
    assert opt_summary.clear_span == 3
    assert list(opt_summary.clear_qubits) == [q]
    assert len(opt_summary.new_operations) == 1
    assert isinstance(opt_summary.new_operations[0].gate,
                      cirq.SingleQubitMatrixGate)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary_effect(opt_summary.new_operations[0]),
        cirq.unitary_effect(cirq.Y**0.5),
        atol=1e-7)


def test_removes_identity_sequence():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([cirq.H(q)]),
            cirq.Moment([cirq.X(q)]),
            cirq.Moment([cirq.H(q)]),
        ]),
        expected=cirq.Circuit())


def test_stopped_at_2qubit():
    m = cirq.MergeSingleQubitGates()
    q = cirq.QubitId()
    q2 = cirq.QubitId()
    c = cirq.Circuit([
        cirq.Moment([cirq.Z(q)]),
        cirq.Moment([cirq.H(q)]),
        cirq.Moment([cirq.X(q)]),
        cirq.Moment([cirq.H(q)]),
        cirq.Moment([cirq.CZ(q, q2)]),
        cirq.Moment([cirq.H(q)]),
    ])

    opt_summary = m.optimization_at(c, 0, c.operation_at(q, 0))
    assert opt_summary.clear_span == 4
    assert list(opt_summary.clear_qubits) == [q]
    if len(opt_summary.new_operations) != 0:
        assert len(opt_summary.new_operations) == 1
        assert isinstance(opt_summary.new_operations[0].gate,
                          cirq.SingleQubitMatrixGate)
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary_effect(opt_summary.new_operations[0]),
            np.eye(2),
            atol=1e-7)


def test_ignores_2qubit_target():
    m = cirq.MergeSingleQubitGates()
    q = cirq.QubitId()
    q2 = cirq.QubitId()
    c = cirq.Circuit([
        cirq.Moment([cirq.CZ(q, q2)]),
    ])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    assert c == cirq.Circuit([cirq.Moment([cirq.CZ(q, q2)])])


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.Gate):
        pass

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(
        UnsupportedDummy()(q0),
    )
    c_orig = cirq.Circuit(circuit)
    cirq.MergeSingleQubitGates().optimize_circuit(circuit)

    assert circuit == c_orig
