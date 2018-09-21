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
import cirq.google as cg


def test_sensitive_to_phase():
    q = cirq.NamedQubit('q')

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([])
        ]),
        cirq.Circuit(),
        atol=0)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.Z(q)**0.0001])
            ]),
            cirq.Circuit(),
            atol=0)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.0001])
        ]),
        cirq.Circuit(),
        atol=0.01)


def test_sensitive_to_measurement_but_not_measured_phase():
    q = cirq.NamedQubit('q')

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit(),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(q)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([cirq.measure(q)]),
        ]),
        atol=1e-8)


def test_sensitive_to_measurement_toggle():
    q = cirq.NamedQubit('q')

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.X(q)]),
                cirq.Moment([cirq.measure(q)]),
            ]),
            atol=1e-8)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.measure(q, invert_mask=(True,))]),
            ]),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(q)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.X(q)]),
            cirq.Moment([cirq.measure(q, invert_mask=(True,))]),
        ]),
        atol=1e-8)


def test_measuring_qubits():
    a, b = cirq.LineQubit.range(2)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(a)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.measure(b)])
            ]),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b, invert_mask=(True,))])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.measure(b, a, invert_mask=(False, True))])
        ]),
        atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a)]),
            cirq.Moment([cirq.measure(b)]),
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b)])
        ]),
        atol=1e-8)


@pytest.mark.parametrize(
    'circuit',
    [
        cirq.testing.random_circuit(cirq.LineQubit.range(2), 4, 0.5)
        for _ in range(5)
    ]
)
def test_random_same_matrix(circuit):
    a, b = cirq.LineQubit.range(2)
    same = cirq.Circuit.from_ops(
        cirq.TwoQubitMatrixGate(circuit.to_unitary_matrix(
            qubits_that_should_be_present=[a, b])).on(a, b))

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, same, atol=1e-8)

    circuit.append(cirq.measure(a))
    same.append(cirq.measure(a))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, same, atol=1e-8)


def test_correct_qubit_ordering():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit.from_ops(cirq.Z(a),
                              cirq.Z(b),
                              cirq.measure(b)),
        cirq.Circuit.from_ops(cirq.Z(a),
                              cirq.measure(b)),
        atol=1e-8)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit.from_ops(cirq.Z(a),
                                  cirq.Z(b),
                                  cirq.measure(b)),
            cirq.Circuit.from_ops(cirq.Z(b),
                                  cirq.measure(b)),
            atol=1e-8)


def test_known_old_failure():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        actual=cirq.Circuit.from_ops(
            cg.ExpWGate(half_turns=0.61351656,
                        axis_half_turns=0.8034575038876517).on(b),
            cirq.measure(a, b)),
        reference=cirq.Circuit.from_ops(
            cg.ExpWGate(half_turns=0.61351656,
                        axis_half_turns=0.8034575038876517).on(b),
            cg.ExpZGate(half_turns=0.5).on(a),
            cg.ExpZGate(half_turns=0.1).on(b),
            cirq.measure(a, b)),
        atol=1e-8)


def test_assert_same_circuits():
    a, b = cirq.LineQubit.range(2)

    cirq.testing.assert_same_circuits(
        cirq.Circuit.from_ops(cirq.H(a)),
        cirq.Circuit.from_ops(cirq.H(a)),
    )

    with pytest.raises(AssertionError, match='differing moment:\n0\n'):
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.H(a)),
            cirq.Circuit(),
        )

    with pytest.raises(AssertionError, match='differing moment:\n1\n'):
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.H(a), cirq.H(a)),
            cirq.Circuit.from_ops(cirq.H(a), cirq.CZ(a, b)),
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.CNOT(a, b)),
            cirq.Circuit.from_ops(cirq.ControlledGate(cirq.X).on(a, b)),
        )
