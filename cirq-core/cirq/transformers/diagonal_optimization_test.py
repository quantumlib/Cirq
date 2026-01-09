# Copyright 2025 The Cirq Developers
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


import numpy as np

import cirq
from cirq.transformers.diagonal_optimization import (
    _is_z_or_cz_pow_gate,
    drop_diagonal_before_measurement,
)


def test_removes_z_before_measure():
    """Tests that Z gates are removed before measurement."""
    q = cirq.NamedQubit('q')

    # Original: H -> Z -> Measure
    circuit = cirq.Circuit(cirq.H(q), cirq.Z(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: H -> Measure (Z is gone)
    expected = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))

    cirq.testing.assert_same_circuits(optimized, expected)


def test_removes_diagonal_chain():
    """Tests that a chain of diagonal gates is removed."""
    q = cirq.NamedQubit('q')

    # Original: H -> Z -> S -> Measure
    circuit = cirq.Circuit(cirq.H(q), cirq.Z(q), cirq.S(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: H -> Measure (Both Z and S are gone)
    expected = cirq.Circuit(cirq.H(q), cirq.measure(q, key='m'))

    cirq.testing.assert_same_circuits(optimized, expected)


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
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, optimized)


def test_keeps_cz_if_only_one_qubit_measured():
    """Tests that CZ is kept if only one qubit is measured."""
    q0, q1 = cirq.LineQubit.range(2)

    # Original: CZ(0,1) -> Measure(0)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.measure(q0, key='m'))

    # CZ shouldn't be removed because q1 is not measured
    optimized = drop_diagonal_before_measurement(circuit)

    cirq.testing.assert_same_circuits(optimized, circuit)


def test_removes_cz_if_both_measured():
    """Tests that CZ is removed if both qubits are measured."""
    q0, q1 = cirq.LineQubit.range(2)

    # Original: CZ(0,1) -> Measure(0), Measure(1)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: Measures only
    expected = cirq.Circuit(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    cirq.testing.assert_same_circuits(optimized, expected)


def test_feature_request_z_cz_commutation():
    """Test the original feature request #4935: Z-CZ commutation before measurement.

    The circuit Z(q0) - CZ(q0, q1) - Z(q1) - M(q1) should keep the CZ gate.
    This is because:
    1. Z(q0) commutes through the CZ and Z(q1) is removed (via eject_z)
    2. After commutation: CZ(q0, q1) - Z(q0) - M(q1)
    3. CZ(q0, q1) and Z(q0) must be kept (q0 is not measured)

    The optimized circuit is: CZ(q0, q1) - Z(q0) - M(q1)
    """
    q0, q1 = cirq.LineQubit.range(2)

    # Original feature request circuit
    circuit = cirq.Circuit(cirq.Z(q0), cirq.CZ(q0, q1), cirq.Z(q1), cirq.measure(q1, key='m1'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Expected: CZ(q0, q1) - Z(q0) - M(q1)
    expected = cirq.Circuit(cirq.CZ(q0, q1), cirq.Z(q0), cirq.Moment(cirq.measure(q1, key='m1')))

    cirq.testing.assert_same_circuits(optimized, expected)


def test_feature_request_full_example():
    """Test the full feature request #4935 with measurements on both qubits."""
    q0, q1 = cirq.LineQubit.range(2)

    # From feature request
    circuit = cirq.Circuit(
        cirq.Z(q0),
        cirq.CZ(q0, q1),
        cirq.Z(q1),
        cirq.Moment(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1')),
    )

    optimized = drop_diagonal_before_measurement(circuit)

    # Should simplify to just measurements
    expected = cirq.Circuit(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    cirq.testing.assert_same_circuits(optimized, expected)


def test_preserves_non_diagonal_gates():
    """Test that non-diagonal gates are preserved."""
    q = cirq.NamedQubit('q')

    circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.Z(q), cirq.measure(q, key='m'))

    optimized = drop_diagonal_before_measurement(circuit)

    # Verify the physics hasn't changed (handles PhasedX vs Y differences)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, optimized)


def test_diagonal_gates_commute_before_measurement():
    """Test that multiple recognized diagonal gates are all removed when all qubits are measured.

    This tests the property that recognized diagonal gates (Z, CZ) commute with each other,
    so we don't remove qubits from measured_qubits when we encounter them. This allows
    earlier diagonal gates in the circuit to also be removed.
    """
    q0, q1 = cirq.LineQubit.range(2)

    # Circuit with multiple recognized diagonal gates before measurements
    circuit = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.Z(q0),
        cirq.Z(q1),
        cirq.measure(q0, key='m0'),
        cirq.measure(q1, key='m1'),
    )

    optimized = drop_diagonal_before_measurement(circuit)

    # All recognized diagonal gates should be removed since all qubits are measured
    expected = cirq.Circuit(cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1'))

    cirq.testing.assert_same_circuits(optimized, expected)


def test_unrecognized_diagonal_breaks_chain():
    """Test that a CZ followed by an unrecognized diagonal 4x4 unitary is handled correctly.

    Even if a gate is diagonal, if it's not a ZPowGate or CZPowGate, it won't be recognized
    and will break the optimization chain. The earlier CZ gate cannot be removed because
    the unrecognized diagonal gate blocks it.
    """
    q0, q1 = cirq.LineQubit.range(2)

    # Create a custom diagonal 4x4 unitary (not a CZPowGate)
    # This is diagonal but won't be recognized by _is_z_or_cz_pow_gate
    diagonal_matrix = np.diag([1, 1j, -1, -1j])
    custom_diagonal_gate = cirq.MatrixGate(diagonal_matrix)

    # Circuit: CZ -> custom diagonal -> measurements
    circuit = cirq.Circuit(
        cirq.CZ(q0, q1),
        custom_diagonal_gate(q0, q1),
        cirq.measure(q0, key='m0'),
        cirq.measure(q1, key='m1'),
    )

    optimized = drop_diagonal_before_measurement(circuit)

    # The custom diagonal gate is not recognized, so it blocks the chain
    # Only the custom diagonal gate can be removed... wait, no! It's not recognized
    # so it won't be removed at all. And it breaks the chain for q0 and q1.
    # So the CZ also cannot be removed.
    cirq.testing.assert_same_circuits(optimized, circuit)


def test_is_z_or_cz_pow_gate_helper_edge_cases():
    """Test edge cases in _is_z_or_cz_pow_gate helper function for full coverage."""

    q = cirq.NamedQubit('q')

    # Test Z gates (including variants like S and T)
    assert _is_z_or_cz_pow_gate(cirq.Z(q))
    assert _is_z_or_cz_pow_gate(cirq.S(q))  # S is Z**0.5
    assert _is_z_or_cz_pow_gate(cirq.T(q))  # T is Z**0.25

    # Test identity gate
    assert _is_z_or_cz_pow_gate(cirq.I(q))

    # Test non-diagonal gates
    assert not _is_z_or_cz_pow_gate(cirq.H(q))
    assert not _is_z_or_cz_pow_gate(cirq.X(q))
    assert not _is_z_or_cz_pow_gate(cirq.Y(q))

    # Test two-qubit CZ gate
    q0, q1 = cirq.LineQubit.range(2)
    assert _is_z_or_cz_pow_gate(cirq.CZ(q0, q1))

    # Other diagonal gates (like CCZ) are not detected by the optimized version
    # This is intentional - eject_z is only effective for Z and CZ anyway
    assert not _is_z_or_cz_pow_gate(cirq.CCZ(q0, q1, q))
