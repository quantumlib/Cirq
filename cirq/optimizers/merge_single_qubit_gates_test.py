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
from typing import Callable, Optional

import numpy as np
import pytest

import cirq


def assert_optimizes(
        before: cirq.Circuit,
        expected: cirq.Circuit,
        optimizer: Optional[Callable[[cirq.Circuit], None]] = None):
    if optimizer is None:
        optimizer = cirq.MergeSingleQubitGates().optimize_circuit
    optimizer(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        cirq.DropNegligible(),
        cirq.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post(before)  # type: ignore #  error: "object" not callable
        post(expected)  # type: ignore #  error: "object" not callable

    assert before == expected, 'BEFORE:\n{}\nEXPECTED:\n{}'.format(
        before, expected)


def test_leaves_singleton():
    m = cirq.MergeSingleQubitGates()
    q = cirq.NamedQubit('q')
    c = cirq.Circuit([cirq.Moment([cirq.X(q)])])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([cirq.Moment([cirq.X(q)])]))


def test_not_both():
    with pytest.raises(ValueError):
        _ = cirq.MergeSingleQubitGates(
            synthesizer=lambda *args: None,
            rewriter=lambda *args: None)


def test_combines_sequence():
    m = cirq.MergeSingleQubitGates()
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(cirq.X(q)**0.5, cirq.Z(q)**0.5, cirq.X(q)**-0.5)

    opt_summary = m.optimization_at(c, 0, c.operation_at(q, 0))
    assert opt_summary.clear_span == 3
    assert list(opt_summary.clear_qubits) == [q]
    assert len(opt_summary.new_operations) == 1
    assert isinstance(opt_summary.new_operations[0].gate, cirq.MatrixGate)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(opt_summary.new_operations[0]),
        cirq.unitary(cirq.Y**0.5),
        atol=1e-7)


def test_removes_identity_sequence():
    q = cirq.NamedQubit('q')
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
    q = cirq.NamedQubit('q')
    q2 = cirq.NamedQubit('q2')
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
        assert isinstance(opt_summary.new_operations[0].gate, cirq.MatrixGate)
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(opt_summary.new_operations[0]),
            np.eye(2),
            atol=1e-7)


def test_ignores_2qubit_target():
    m = cirq.MergeSingleQubitGates()
    q = cirq.NamedQubit('q')
    q2 = cirq.NamedQubit('q2')
    c = cirq.Circuit([
        cirq.Moment([cirq.CZ(q, q2)]),
    ])

    m.optimization_at(c, 0, c.operation_at(q, 0))

    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit([cirq.Moment([cirq.CZ(q, q2)])]))


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.SingleQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(UnsupportedDummy()(q0),)
    c_orig = cirq.Circuit(circuit)
    cirq.MergeSingleQubitGates().optimize_circuit(circuit)

    assert circuit == c_orig


def test_rewrite():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q1),
        cirq.Y(q0),
        cirq.CZ(q0, q1),
        cirq.Y(q1),
    )
    cirq.MergeSingleQubitGates(
        rewriter=lambda ops: cirq.H(ops[0].qubits[0])
    ).optimize_circuit(circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit)

    cirq.testing.assert_same_circuits(
        circuit,
        cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.CZ(q0, q1),
            cirq.H(q1),
        ))


def test_merge_single_qubit_gates_into_phased_x_z():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            cirq.X(a),
            cirq.Y(b)**0.5,
            cirq.CZ(a, b),
            cirq.H(a),
            cirq.Z(a),
        ),
        expected=cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=1)(a),
            cirq.Y(b)**0.5,
            cirq.CZ(a, b),
            (cirq.PhasedXPowGate(phase_exponent=-0.5)(a))**0.5,
        ),
        optimizer=cirq.merge_single_qubit_gates_into_phased_x_z,
    )


def test_merge_single_qubit_gates_into_phxz():

    def phxz(a, x, z):
        return cirq.PhasedXZGate(
            axis_phase_exponent=a,
            x_exponent=x,
            z_exponent=z,
        )

    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            cirq.X(a),
            cirq.Y(b)**0.5,
            cirq.CZ(a, b),
            cirq.H(a),
            cirq.Z(a),
        ),
        expected=cirq.Circuit(
            phxz(-1, 1, 0).on(a),
            phxz(0.5, 0.5, 0).on(b),
            cirq.CZ(a, b),
            phxz(-0.5, 0.5, 0).on(a),
        ),
        optimizer=cirq.merge_single_qubit_gates_into_phxz,
    )
