# Copyright 2022 The Cirq Developers
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
from typing import List

import numpy as np
import pytest

import cirq


def assert_optimizes(optimized: cirq.AbstractCircuit, expected: cirq.AbstractCircuit):
    # Ignore differences that would be caught by follow-up optimizations.
    followup_transformers: List[cirq.TRANSFORMER] = [
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]
    for transform in followup_transformers:
        optimized = transform(optimized)
        expected = transform(expected)

    cirq.testing.assert_same_circuits(optimized, expected)


def test_not_both():
    with pytest.raises(ValueError, match='both rewriter and synthesizer'):
        _ = cirq.merge_single_qubit_gates(
            cirq.Circuit(), synthesizer=lambda *args: None, rewriter=lambda *args: None
        )


def test_combines_sequence():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(cirq.X(q) ** 0.5, cirq.Z(q) ** 0.5, cirq.X(q) ** -0.5)
    c = cirq.merge_single_qubit_gates(c)
    op_list = [*c.all_operations()]
    assert len(op_list) == 1
    assert isinstance(op_list[0].gate, cirq.MatrixGate)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(c), cirq.unitary(cirq.Y ** 0.5), atol=1e-7
    )


def test_removes_identity_sequence():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates(
            cirq.Circuit([cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q)])
        ),
        expected=cirq.Circuit(),
    )


def test_stopped_at_2qubit():
    q, q2 = cirq.LineQubit.range(2)
    c = cirq.Circuit([cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.CZ(q, q2), cirq.H(q)])
    c = cirq.drop_empty_moments(cirq.merge_single_qubit_gates(c))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-7)
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)


def test_respects_nocompile_tags():
    q = cirq.NamedQubit("q")
    c = cirq.Circuit(
        [cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.X(q).with_tags("nocompile"), cirq.H(q)]
    )
    context = cirq.TransformerContext(tags_to_ignore=("nocompile",))
    c = cirq.drop_empty_moments(cirq.merge_single_qubit_gates(c, context=context))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-7)
    assert c[1][q] == cirq.X(q).with_tags("nocompile")
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)


def test_ignores_2qubit_target():
    c = cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)))
    assert_optimizes(optimized=cirq.merge_single_qubit_gates(c), expected=c)


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.SingleQubitGate):
        pass

    c = cirq.Circuit(UnsupportedDummy()(cirq.LineQubit(0)))
    assert_optimizes(optimized=cirq.merge_single_qubit_gates(c), expected=c)


def test_rewrite():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.X(q1), cirq.CZ(q0, q1), cirq.Y(q1))
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates(
            circuit, rewriter=lambda ops: cirq.H(ops[0].qubits[0])
        ),
        expected=cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.CZ(q0, q1), cirq.H(q1)),
    )


def test_merge_single_qubit_gates_into_phased_x_z():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.Y(b) ** 0.5, cirq.CZ(a, b), cirq.H(a), cirq.Z(a))
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phased_x_and_z(c),
        expected=cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=1)(a),
            cirq.Y(b) ** 0.5,
            cirq.CZ(a, b),
            (cirq.PhasedXPowGate(phase_exponent=-0.5)(a)) ** 0.5,
        ),
    )


def test_merge_single_qubit_gates_into_phxz():
    def phxz(a, x, z):
        return cirq.PhasedXZGate(
            axis_phase_exponent=a,
            x_exponent=x,
            z_exponent=z,
        )

    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.Y(b) ** 0.5, cirq.CZ(a, b), cirq.H(a), cirq.Z(a))
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phxz(c),
        expected=cirq.Circuit(
            phxz(-1, 1, 0).on(a), phxz(0.5, 0.5, 0).on(b), cirq.CZ(a, b), phxz(-0.5, 0.5, 0).on(a)
        ),
    )
