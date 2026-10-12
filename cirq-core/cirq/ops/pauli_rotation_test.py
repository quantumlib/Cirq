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


def test_raises_for_invalid_pauli_string() -> None:
    with pytest.raises(TypeError, match='DensePauliString'):
        cirq.PauliRotation(cirq.X)  # type: ignore[arg-type]


@pytest.mark.parametrize('pauli_string', (cirq.X(q0) * cirq.Z(q1), 'XZ'))
def test_raises_for_non_dense_pauli_string(pauli_string) -> None:
    with pytest.raises(TypeError, match='DensePauliString'):
        cirq.PauliRotation(pauli_string, 0.5, [q0, q1])


def test_raises_for_wrong_number_of_qubits() -> None:
    with pytest.raises(ValueError, match='Expected 2 qubits'):
        cirq.PauliRotation(cirq.DensePauliString('XZ'), qubits=[q0])


def test_raises_for_no_qubits_supplied() -> None:
    with pytest.raises(ValueError, match='Expected qubits'):
        cirq.PauliRotation(cirq.DensePauliString('XZ'))


def test_properties() -> None:
    rot = cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.5, [q0, q1, q2])
    assert isinstance(rot, cirq.GateOperation)
    assert rot.pauli_string == cirq.DensePauliString('XIZ')
    assert isinstance(rot.pauli_string, cirq.DensePauliString)
    assert rot.exponent == 0.5
    assert rot.gate == cirq.PauliRotationGate('XIZ', 0.5)
    assert cirq.PauliRotation(cirq.DensePauliString('XZ'), qubits=[q0, q1]).exponent == 1


def test_qubits_keep_identities() -> None:
    rot = cirq.PauliRotation(cirq.DensePauliString('XIZ'), qubits=[q3, q2, q1])
    assert rot.qubits == (q3, q2, q1)


def test_with_qubits() -> None:
    rot = cirq.PauliRotation(cirq.DensePauliString('XIY'), 0.3, qubits=[q0, q1, q2])
    assert rot.with_qubits(q1, q2, q3) == cirq.PauliRotation(
        cirq.DensePauliString('XIY'), 0.3, qubits=[q1, q2, q3]
    )


def test_with_gate() -> None:
    rot = cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.3, [q0, q2])
    assert rot.with_gate(rot.gate) is rot
    assert rot.with_gate(cirq.PauliRotationGate('YY', 0.5)) == cirq.PauliRotation(
        cirq.DensePauliString('YY'), 0.5, [q0, q2]
    )
    assert rot.with_gate(cirq.CZ) == cirq.CZ(q0, q2)


def test_parameterized() -> None:
    t = sympy.Symbol('t')
    rot = cirq.PauliRotation(cirq.DensePauliString('XZ'), t, [q0, q1])
    assert cirq.is_parameterized(rot)
    assert cirq.parameter_names(rot) == {'t'}
    assert not cirq.has_unitary(rot)
    with pytest.raises(ValueError, match='parameterized'):
        rot.matrix()
    resolved = cirq.resolve_parameters(rot, {'t': 0.5})
    assert resolved == cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.5, [q0, q1])
    assert not cirq.is_parameterized(resolved)


def test_pow() -> None:
    rot = cirq.PauliRotation(cirq.DensePauliString('XY'), 0.3, qubits=[q2, q3])
    assert rot**2 == cirq.PauliRotation(cirq.DensePauliString('XY'), 0.6, qubits=[q2, q3])
    np.testing.assert_allclose(cirq.unitary(rot**2), cirq.unitary(rot) @ cirq.unitary(rot))
    assert rot**-1 == cirq.PauliRotation(cirq.DensePauliString('XY'), -0.3, qubits=[q2, q3])


@pytest.mark.parametrize(
    'rot',
    (
        cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.3, qubits=[q1, q2, q3]),
        cirq.PauliRotation(cirq.DensePauliString('X'), sympy.Symbol('t'), qubits=[q0]),
        cirq.PauliRotation(cirq.DensePauliString('XZ', coefficient=-1), 0.3, qubits=[q2, q3]),
    ),
)
def test_repr(rot) -> None:
    cirq.testing.assert_equivalent_repr(rot)


def test_str() -> None:
    assert (
        str(cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.5, [q0, q1, q2]))
        == 'exp(j * 0.5 * (+XIZ)) on q(0), q(1), q(2)'
    )


def test_equality() -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.5, [q0, q1]),
        cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.5, [q0, q1]),
        cirq.PauliRotationGate(cirq.DensePauliString('XZ'), 0.5).on(q0, q1),
    )
    eq.add_equality_group(cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.25, [q0, q1]))
    eq.add_equality_group(cirq.PauliRotation(cirq.DensePauliString('ZX'), 0.5, [q1, q0]))
    eq.add_equality_group(cirq.PauliRotation(cirq.DensePauliString('XZ'), 0.5, [q1, q2]))
    eq.add_equality_group(
        cirq.PauliRotation(cirq.DensePauliString('XZ', coefficient=-1), 0.5, [q0, q1])
    )


def test_created_by_gate_on() -> None:
    gate = cirq.PauliRotationGate('XIZ', 0.3)
    op = gate.on(q1, q2, q3)
    assert isinstance(op, cirq.PauliRotation)
    assert op == cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.3, [q1, q2, q3])
    assert op.gate == gate
    assert gate(q1, q2, q3) == op


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
def test_unitary(pauli, k) -> None:
    op = cirq.PauliRotation(pauli, k, [q0, q1, q2][: len(pauli)])
    assert cirq.has_unitary(op)
    np.testing.assert_allclose(cirq.unitary(op), _expected_unitary(pauli, k), atol=1e-8)
    cirq.testing.assert_decompose_is_consistent_with_unitary(op)


def test_decomposes_to_pauli_string_phasor() -> None:
    (op,) = cirq.decompose_once(
        cirq.PauliRotation(cirq.DensePauliString('Y', coefficient=-1), 0.3, [q0])
    )
    assert op == cirq.PauliStringPhasor(
        cirq.Y(q0), exponent_neg=0.3 / np.pi, exponent_pos=-0.3 / np.pi
    )


@pytest.mark.parametrize(
    'pauli, k, qubits',
    (
        (cirq.DensePauliString('XIZ'), 0.4, [q0, q1, q2]),
        (cirq.DensePauliString('XIZ'), 0.4, [q2, q1, q0]),
        (cirq.DensePauliString('IXI'), np.pi / 4, [q3, q0, q2]),
        (cirq.DensePauliString('IYZ', coefficient=-1), 1.2, [q1, q3, q0]),
        (cirq.DensePauliString('III', coefficient=3), 0.2, [q0, q1, q2]),
    ),
)
def test_decomposition_unitary_with_identities(pauli, k, qubits) -> None:
    rot = cirq.PauliRotation(pauli, k, qubits)
    decomposition = cirq.decompose_once(rot)
    acted_on = {q for op in decomposition for q in op.qubits}
    assert acted_on == {q for q, p in zip(qubits, pauli) if p != cirq.I}
    actual = cirq.Circuit(decomposition).unitary(qubit_order=qubits)
    np.testing.assert_allclose(actual, _expected_unitary(pauli, k), atol=1e-8)


@pytest.mark.parametrize(
    'pauli, k',
    ((cirq.DensePauliString('X', coefficient=1j), 0.5), (cirq.DensePauliString('X'), 0.5j)),
)
def test_no_unitary_for_non_real_angle(pauli, k) -> None:
    op = cirq.PauliRotation(pauli, k, [q0])
    assert not cirq.has_unitary(op)
    assert cirq.unitary(op, None) is None
    with pytest.raises(ValueError, match='imaginary'):
        op.matrix()
