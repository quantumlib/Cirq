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

import numpy as np
import pytest
import sympy

import cirq

_ATOL = 1e-8


def _expected_pauli_rotation_unitary(pauli_label: str, exponent: float) -> np.ndarray:
    """Reference unitary for U = cos(theta)I + i sin(theta) P."""
    pauli = cirq.unitary(cirq.DensePauliString(pauli_label))
    cos_theta = np.cos(exponent)
    sin_theta = np.sin(exponent)
    identity = np.eye(pauli.shape[0], dtype=complex)
    return cos_theta * identity + 1j * sin_theta * pauli


@pytest.mark.parametrize(
    'pauli_label,exponent',
    [
        ('X', 0.0),
        ('X', np.pi / 7),
        ('Y', np.pi / 3),
        ('Z', -np.pi / 5),
        ('XI', np.pi / 4),
        ('IX', np.pi / 6),
        ('XY', np.pi / 8),
        ('ZZ', np.pi / 2),
        ('XYZ', np.pi / 9),
    ],
)
def test_pauli_rotation_unitary_matches_analytic_formula(pauli_label: str, exponent: float) -> None:
    qubits = cirq.LineQubit.range(len(pauli_label))
    op = cirq.PauliRotation(pauli_label, qubits, exponent=exponent)
    expected = _expected_pauli_rotation_unitary(pauli_label, exponent)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(op), expected, atol=_ATOL)


def test_pauli_rotation_differs_from_pauli_sum_exponential() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    theta = np.pi / 4
    rotation = cirq.unitary(cirq.PauliRotation('XI', [q0, q1], exponent=theta))
    wrong = cirq.unitary(
        cirq.PauliSumExponential(cirq.DensePauliString('XI')(*[q0, q1]), exponent=theta)
    )
    assert rotation.shape == (4, 4)
    assert wrong.shape == (2, 2)
    assert not np.allclose(rotation[:2, :2], wrong)


@pytest.mark.parametrize(
    'pauli_label,exponent', [('X', np.pi / 5), ('YZ', np.pi / 6), ('XY', np.pi / 8)]
)
def test_pauli_rotation_decomposition_matches_unitary(pauli_label: str, exponent: float) -> None:
    qubits = cirq.LineQubit.range(len(pauli_label))
    op = cirq.PauliRotation(pauli_label, qubits, exponent=exponent)
    decomposed = cirq.Circuit(cirq.decompose(op))
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(op), cirq.unitary(decomposed), atol=_ATOL
    )


def test_pauli_rotation_repr_roundtrip() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    cirq.testing.assert_equivalent_repr(cirq.PauliRotation('XI', [q0, q1], exponent=np.pi / 4))
    cirq.testing.assert_equivalent_repr(
        cirq.PauliRotationGate(cirq.DensePauliString('XI'), exponent=np.pi / 4)
    )


def test_pauli_rotation_parameter_resolution() -> None:
    theta = sympy.Symbol('theta')
    q0, q1 = cirq.LineQubit.range(2)
    gate = cirq.PauliRotationGate(cirq.DensePauliString('XI'), exponent=theta)
    op = cirq.PauliRotation('XI', [q0, q1], exponent=theta)

    assert cirq.is_parameterized(gate)
    assert cirq.is_parameterized(op)
    assert 'theta' in gate._parameter_names_()

    resolved_angle = np.pi / 11
    resolver = cirq.ParamResolver({'theta': resolved_angle})
    resolved_gate = cirq.resolve_parameters(gate, resolver)
    resolved_op = cirq.resolve_parameters(op, resolver)

    assert not cirq.is_parameterized(resolved_gate)
    assert not cirq.is_parameterized(resolved_op)
    assert resolved_gate.exponent == resolved_angle
    assert resolved_op.gate.exponent == resolved_angle

    expected = _expected_pauli_rotation_unitary('XI', resolved_angle)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(resolved_op), expected, atol=_ATOL)


def test_pauli_rotation_gate_rejects_non_unit_coefficient() -> None:
    dps = cirq.DensePauliString('X') * 2
    with pytest.raises(ValueError, match='unit Pauli string'):
        cirq.PauliRotationGate(dps, exponent=0.1)


def test_pauli_rotation_rejects_qubit_mismatch() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='must match'):
        cirq.PauliRotation('X', [q0, q1], exponent=0.1)


def test_pauli_rotation_gate_unitary_not_implemented_when_parameterized() -> None:
    theta = sympy.Symbol('theta')
    gate = cirq.PauliRotationGate(cirq.DensePauliString('X'), exponent=theta)
    assert cirq.unitary(gate, default=None) is None


def test_pauli_rotation_gate_decompose_not_implemented_when_parameterized() -> None:
    theta = sympy.Symbol('theta')
    gate = cirq.PauliRotationGate(cirq.DensePauliString('X'), exponent=theta)
    result = gate._decompose_((cirq.LineQubit(0),))
    assert result is NotImplemented


def test_pauli_rotation_gate_circuit_diagram_info() -> None:
    gate = cirq.PauliRotationGate(cirq.DensePauliString('X'), exponent=0.5)
    info = gate._circuit_diagram_info_(cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    assert 'PR(' in str(info)


def test_pauli_rotation_gate_pow() -> None:
    gate = cirq.PauliRotationGate(cirq.DensePauliString('X'), exponent=np.pi / 4)
    doubled = gate**2
    assert doubled == cirq.PauliRotationGate(cirq.DensePauliString('X'), exponent=np.pi / 2)


def test_pauli_rotation_pow() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.PauliRotation('XI', [q0, q1], exponent=np.pi / 4)
    doubled = op**2
    expected = cirq.PauliRotation('XI', [q0, q1], exponent=np.pi / 2)
    assert doubled == expected
