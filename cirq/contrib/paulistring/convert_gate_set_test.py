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
from cirq.contrib.paulistring import PauliStringPhasor

from cirq.contrib.paulistring import converted_gate_set


@pytest.mark.parametrize('op,expected_ops', (lambda q0, q1: (
    (cirq.X(q0), cirq.CliffordGate.X(q0)),
    (cirq.Y(q0), cirq.CliffordGate.Y(q0)),
    (cirq.Z(q0), cirq.CliffordGate.Z(q0)),
    (cirq.X(q0) ** 0.5, cirq.CliffordGate.X_sqrt(q0)),
    (cirq.Y(q0) ** 0.5, cirq.CliffordGate.Y_sqrt(q0)),
    (cirq.Z(q0) ** 0.5, cirq.CliffordGate.Z_sqrt(q0)),
    (cirq.X(q0) ** -0.5, cirq.CliffordGate.X_nsqrt(q0)),
    (cirq.Y(q0) ** -0.5, cirq.CliffordGate.Y_nsqrt(q0)),
    (cirq.Z(q0) ** -0.5, cirq.CliffordGate.Z_nsqrt(q0)),

    (cirq.X(q0) ** 0.25,
     PauliStringPhasor(cirq.PauliString.from_single(q0, cirq.Pauli.X)) ** 0.25),
    (cirq.Y(q0) ** 0.25,
     PauliStringPhasor(cirq.PauliString.from_single(q0, cirq.Pauli.Y)) ** 0.25),
    (cirq.Z(q0) ** 0.25,
     PauliStringPhasor(cirq.PauliString.from_single(q0, cirq.Pauli.Z)) ** 0.25),

    (cirq.X(q0) ** 0, ()),

    (cirq.CZ(q0, q1),
     cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.Z, False
                                  )(q0, q1)),

    (cirq.MeasurementGate('key')(q0, q1),
     cirq.MeasurementGate('key')(q0, q1)),
))(cirq.LineQubit(0), cirq.LineQubit(1)))
def test_converts_various_ops(op, expected_ops):
    before = cirq.Circuit.from_ops(op)
    expected = cirq.Circuit.from_ops(expected_ops,
                                     strategy=cirq.InsertStrategy.EARLIEST)

    after = converted_gate_set(before)
    assert after == expected
    cirq.testing.assert_allclose_up_to_global_phase(
            before.to_unitary_matrix(),
            after.to_unitary_matrix(qubits_that_should_be_present=op.qubits),
            atol=1e-7)
    cirq.testing.assert_allclose_up_to_global_phase(
            after.to_unitary_matrix(qubits_that_should_be_present=op.qubits),
            expected.to_unitary_matrix(qubits_that_should_be_present=op.qubits),
            atol=1e-7)


def test_converts_single_qubit_series():
    q0, = cirq.LineQubit.range(1)

    before = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.X(q0) ** 0.5,
        cirq.Y(q0) ** 0.5,
        cirq.Z(q0) ** 0.5,
        cirq.X(q0) ** -0.5,
        cirq.Y(q0) ** -0.5,
        cirq.Z(q0) ** -0.5,
        cirq.X(q0) ** 0.25,
        cirq.Y(q0) ** 0.25,
        cirq.Z(q0) ** 0.25,
    )

    after = converted_gate_set(before)
    cirq.testing.assert_allclose_up_to_global_phase(
            before.to_unitary_matrix(),
            after.to_unitary_matrix(),
            atol=1e-7)


def test_converts_single_qubit_then_two():
    q0, q1 = cirq.LineQubit.range(2)

    before = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.CZ(q0, q1),
    )

    after = converted_gate_set(before)
    cirq.testing.assert_allclose_up_to_global_phase(
            before.to_unitary_matrix(),
            after.to_unitary_matrix(),
            atol=1e-7)


def test_converts_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)

    before = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.X(q0) ** 0.5,
        cirq.Y(q0) ** 0.5,
        cirq.Z(q0) ** 0.5,
        cirq.X(q0) ** -0.5,
        cirq.Y(q0) ** -0.5,
        cirq.Z(q0) ** -0.5,
        cirq.H(q0),
        cirq.CZ(q0, q1),
        cirq.CZ(q1, q2),
        cirq.X(q0) ** 0.25,
        cirq.Y(q0) ** 0.25,
        cirq.Z(q0) ** 0.25,

        cirq.CZ(q0, q1),
    )

    after = converted_gate_set(before)

    cirq.testing.assert_allclose_up_to_global_phase(
            before.to_unitary_matrix(),
            after.to_unitary_matrix(),
            atol=1e-7)

    assert after.to_text_diagram() == '''
0: ───Y^0.5───@───[Z]^-0.304───[X]^0.333───[Z]^0.304───@───[Z]^0.142───
              │                                        │
1: ───────────@───@────────────────────────────────────@───────────────
                  │
2: ───────────────@────────────────────────────────────────────────────
'''.strip()

