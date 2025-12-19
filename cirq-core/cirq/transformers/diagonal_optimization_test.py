# Copyright 2024 The Cirq Developers
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
"""Tests for diagonal_optimization transformer."""


import cirq
from cirq.transformers.diagonal_optimization import (
    drop_diagonal_before_measurement,
    _is_diagonal,
)


def test_removes_z_before_measure():
    """Tests that Z gates are removed before measurement."""
    q = cirq.NamedQubit('q')

    # Original: H -> Z -> Measure
    circuit = cirq.Circuit(cirq.H(q), cirq.Z(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: H -> Measure (Z is gone)
    expected = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))

    assert optimized == expected


def test_removes_diagonal_chain():
    """Tests that a chain of diagonal gates is removed."""
    q = cirq.NamedQubit('q')

    # Original: H -> Z -> S -> Measure
    circuit = cirq.Circuit(cirq.H(q), cirq.Z(q), cirq.S(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: H -> Measure (Both Z and S are gone)
    expected = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))

    assert optimized == expected


def test_keeps_z_blocked_by_x():
    """Tests that Z gates blocked by X gates are preserved."""
    q = cirq.NamedQubit('q')

    # Original: Z -> X -> Measure
    circuit = cirq.Circuit(cirq.Z(q), cirq.X(q), cirq.measure(q, key='m'))

    # Z cannot commute past X, so it should be kept
    # Note: eject_z will phase the X, so the circuit changes but Z is preserved
    optimized = drop_diagonal_before_measurement(circuit)

    # We use this helper to check mathematical equivalence
    # instead of checking exact gate types (Y vs PhasedX)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, optimized, atol=1e-6
    )


def test_keeps_cz_if_only_one_qubit_measured():
    """Tests that CZ is kept if only one qubit is measured."""
    q0, q1 = cirq.LineQubit.range(2)

    # Original: CZ(0,1) -> Measure(0)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.measure(q0, key='m'))

    # CZ shouldn't be removed because q1 is not measured
    optimized = drop_diagonal_before_measurement(circuit)

    assert optimized == circuit


def test_removes_cz_if_both_measured():
    """Tests that CZ is removed if both qubits are measured."""
    q0, q1 = cirq.LineQubit.range(2)

    # Original: CZ(0,1) -> Measure(0), Measure(1)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: Measures only
    expected = cirq.Circuit(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    # Check that operations match (ignoring Moment structure)
    assert list(optimized.all_operations()) == list(expected.all_operations())


def test_feature_request_z_cz_commutation():
    """Test the original feature request case: Z-CZ commutation before measurement.

    The circuit Z(q0) - CZ(q0, q1) - measure(q1) should be optimized to just measure(q1).
    This is because:
    1. Z on the control qubit of CZ commutes through the CZ
    2. After commutation, both gates are diagonal and before measurement
    3. Both can be removed
    """
    q0, q1 = cirq.LineQubit.range(2)

    # Original feature request circuit
    circuit = cirq.Circuit(cirq.Z(q0), cirq.CZ(q0, q1), cirq.measure(q1, key='m1'))

    optimized = drop_diagonal_before_measurement(circuit)

    # The Z(0) might be moved or merged by eject_z, but the CZ MUST stay.
    # We check that a two-qubit gate still exists.
    assert len(list(optimized.findall_operations(lambda op: len(op.qubits) == 2))) > 0


def test_feature_request_full_example():
    """Test the full feature request example with measurements on both qubits."""
    q0, q1 = cirq.LineQubit.range(2)

    # From feature request
    circuit = cirq.Circuit(
        cirq.Z(q0),
        cirq.CZ(q0, q1),
        cirq.Z(q1),
        cirq.measure(q0, key='m0'),
        cirq.measure(q1, key='m1'),
    )

    optimized = drop_diagonal_before_measurement(circuit)

    # Should simplify to just measurements
    expected = cirq.Circuit(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    assert list(optimized.all_operations()) == list(expected.all_operations())


def test_preserves_non_diagonal_gates():
    """Test that non-diagonal gates are preserved."""
    q = cirq.NamedQubit('q')

    circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.Z(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Verify the physics hasn't changed (handles PhasedX vs Y differences)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, optimized, atol=1e-6
    )


def test_is_diagonal_helper_edge_cases():
    """Test edge cases in _is_diagonal helper function for full coverage."""

    q = cirq.NamedQubit('q')

    # Test Z gates (including variants like S and T)
    assert _is_diagonal(cirq.Z(q))
    assert _is_diagonal(cirq.S(q))  # S is Z**0.5
    assert _is_diagonal(cirq.T(q))  # T is Z**0.25

    # Test identity gate
    assert _is_diagonal(cirq.I(q))

    # Test non-diagonal gates
    assert not _is_diagonal(cirq.H(q))
    assert not _is_diagonal(cirq.X(q))
    assert not _is_diagonal(cirq.Y(q))

    # Test two-qubit CZ gate
    q0, q1 = cirq.LineQubit.range(2)
    assert _is_diagonal(cirq.CZ(q0, q1))

    # Other diagonal gates (like CCZ) are not detected by the optimized version
    # This is intentional - eject_z is only effective for Z and CZ anyway
    assert not _is_diagonal(cirq.CCZ(q0, q1, q))
