# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import sympy

import cirq

q0, q1, q2, q3 = cirq.LineQubit.range(4)


def _expected_unitary(pauli: cirq.DensePauliString, k: float) -> np.ndarray:
    unit_pauli = pauli / pauli.coefficient
    return scipy.linalg.expm(1j * k * complex(pauli.coefficient) * cirq.unitary(unit_pauli))


def test_gate_accepts_string() -> None:
    assert cirq.PauliRotationGate('XIZ').dense_pauli_string == cirq.DensePauliString('XIZ')
    assert cirq.PauliRotationGate('XZ', 0.5) == cirq.PauliRotationGate(
        cirq.DensePauliString('XZ'), 0.5
    )


def test_gate_raises_for_invalid_string() -> None:
    with pytest.raises(TypeError):
        cirq.PauliRotationGate('ABC')


def test_gate_num_qubits() -> None:
    assert cirq.num_qubits(cirq.PauliRotationGate('XIZ', 0.3)) == 3


def test_gate_equality() -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PauliRotationGate('XZ', 0.5), cirq.PauliRotationGate(cirq.DensePauliString('XZ'), 0.5)
    )
    eq.add_equality_group(cirq.PauliRotationGate('XZ', 0.25))
    eq.add_equality_group(cirq.PauliRotationGate(cirq.DensePauliString('XZ', coefficient=-1), 0.5))
    eq.add_equality_group(cirq.PauliRotationGate('ZX', 0.5))
    assert cirq.approx_eq(
        cirq.PauliRotationGate('XZ', 0.5), cirq.PauliRotationGate('XZ', 0.5 + 1e-10)
    )


@pytest.mark.parametrize(
    'pauli, k',
    (
        (cirq.DensePauliString('X'), np.pi / 2),
        (cirq.DensePauliString('IX'), np.pi / 4),
        (cirq.DensePauliString('XI'), np.pi / 4),
        (cirq.DensePauliString('XIX'), np.pi / 4),
        (cirq.DensePauliString('IXI'), np.pi / 4),
        (cirq.DensePauliString('XIZ'), 0.4),
        (cirq.DensePauliString('YX', coefficient=-1), 1.2),
        (cirq.DensePauliString('XZ', coefficient=2), 0.7),
        (cirq.DensePauliString('II', coefficient=3), 0.2),
    ),
)
def test_gate_unitary(pauli, k) -> None:
    gate = cirq.PauliRotationGate(pauli, k)
    assert cirq.has_unitary(gate)
    np.testing.assert_allclose(cirq.unitary(gate), _expected_unitary(pauli, k), atol=1e-8)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


def test_gate_decomposes_to_pauli_string_phasor() -> None:
    (op,) = cirq.decompose_once_with_qubits(cirq.PauliRotationGate('XIZ', 0.3), [q0, q1, q2])
    assert op == cirq.PauliStringPhasor(
        cirq.X(q0) * cirq.Z(q2), exponent_neg=-0.3 / np.pi, exponent_pos=0.3 / np.pi
    )


@pytest.mark.parametrize(
    'pauli, k',
    (
        (cirq.DensePauliString('XZ'), 0.3),
        (cirq.DensePauliString('XIZ'), 0.4),
        (cirq.DensePauliString('IXI'), np.pi / 4),
        (cirq.DensePauliString('IIZ'), 1.1),
        (cirq.DensePauliString('YII'), 0.6),
        (cirq.DensePauliString('IYIZI'), 0.7),
        (cirq.DensePauliString('XIY', coefficient=-1), 1.2),
        (cirq.DensePauliString('IZI', coefficient=2), 0.35),
        (cirq.DensePauliString('III', coefficient=3), 0.2),
    ),
)
def test_gate_decomposition_unitary_with_identities(pauli, k) -> None:
    qubits = cirq.LineQubit.range(len(pauli))
    decomposition = cirq.decompose_once_with_qubits(cirq.PauliRotationGate(pauli, k), qubits)
    # Identities are not acted on by the decomposition.
    acted_on = {q for op in decomposition for q in op.qubits}
    assert acted_on == {q for q, p in zip(qubits, pauli) if p != cirq.I}
    # Padding the decomposition with identities on the remaining qubits gives the full rotation.
    actual = cirq.Circuit(decomposition).unitary(qubit_order=qubits)
    assert actual.shape == (2 ** len(pauli), 2 ** len(pauli))
    np.testing.assert_allclose(actual, _expected_unitary(pauli, k), atol=1e-8)


def test_gate_parameterized_decompose() -> None:
    t = sympy.Symbol('t')
    gate = cirq.PauliRotationGate('XZ', t)
    (op,) = cirq.decompose_once_with_qubits(gate, [q0, q1])
    assert isinstance(op, cirq.PauliStringPhasor)
    np.testing.assert_allclose(
        cirq.unitary(cirq.resolve_parameters(op, {'t': 0.5})),
        cirq.unitary(cirq.PauliRotationGate('XZ', 0.5)),
        atol=1e-8,
    )


@pytest.mark.parametrize(
    'pauli, k',
    ((cirq.DensePauliString('X', coefficient=1j), 0.5), (cirq.DensePauliString('X'), 0.5j)),
)
def test_gate_no_unitary_for_non_real_angle(pauli, k) -> None:
    gate = cirq.PauliRotationGate(pauli, k)
    assert not cirq.has_unitary(gate)
    assert cirq.unitary(gate, None) is None
    assert cirq.decompose_once_with_qubits(gate, [q0], None) is None
    with pytest.raises(ValueError, match='imaginary'):
        gate.matrix()


def test_gate_unitary_for_real_angle_from_imaginary_parts() -> None:
    # (1j * X) with exponent -1j gives a real angle of 1.
    gate = cirq.PauliRotationGate(cirq.DensePauliString('X', coefficient=1j), -1j)
    assert cirq.has_unitary(gate)
    np.testing.assert_allclose(
        cirq.unitary(gate), _expected_unitary(cirq.DensePauliString('X'), 1), atol=1e-8
    )
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


def test_gate_parameterized() -> None:
    t = sympy.Symbol('t')
    gate = cirq.PauliRotationGate('XZ', t)
    assert cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == {'t'}
    assert not cirq.has_unitary(gate)
    with pytest.raises(ValueError, match='parameterized'):
        gate.matrix()
    assert cirq.resolve_parameters(gate, {'t': 0.5}) == cirq.PauliRotationGate('XZ', 0.5)
    coefficient_gate = cirq.PauliRotationGate(cirq.DensePauliString('XZ', coefficient=t))
    assert cirq.is_parameterized(coefficient_gate)
    assert cirq.parameter_names(coefficient_gate) == {'t'}


def test_gate_pow() -> None:
    assert cirq.PauliRotationGate('XY', 0.3) ** 2 == cirq.PauliRotationGate('XY', 0.6)


@pytest.mark.parametrize(
    'gate',
    (
        cirq.PauliRotationGate('XIZ', 0.3),
        cirq.PauliRotationGate(cirq.DensePauliString('XZ', coefficient=-1), sympy.Symbol('t')),
    ),
)
def test_gate_repr(gate) -> None:
    cirq.testing.assert_equivalent_repr(gate)


def test_gate_str() -> None:
    assert str(cirq.PauliRotationGate('XIZ', 0.5)) == 'exp(j * 0.5 * (+XIZ))'
