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


def _expected_unitary(pauli: cirq.PauliString | cirq.DensePauliString, k: float) -> np.ndarray:
    unit_pauli = pauli / pauli.coefficient
    return scipy.linalg.expm(1j * k * complex(pauli.coefficient) * cirq.unitary(unit_pauli))


def test_raises_for_invalid_pauli_string() -> None:
    with pytest.raises(TypeError, match='PauliString'):
        cirq.PauliRotation(cirq.X)  # type: ignore[arg-type]


def test_raises_for_qubits_with_pauli_string() -> None:
    with pytest.raises(ValueError, match='with_qubits'):
        cirq.PauliRotation(cirq.X(q0) * cirq.Z(q1), qubits=[q2, q3])


def test_raises_for_wrong_number_of_qubits() -> None:
    with pytest.raises(ValueError, match='Expected 2 qubits'):
        cirq.PauliRotation(cirq.DensePauliString('XZ'), qubits=[q0])


def test_raises_for_no_qubits_supplied_for_dense_pauli_string() -> None:
    with pytest.raises(ValueError):
        cirq.PauliRotation(cirq.DensePauliString('XZ'))


def test_pauli_string_type_is_preserved() -> None:
    sparse = cirq.X(q0) * cirq.Z(q2)
    dense = cirq.DensePauliString('XIZ')
    assert isinstance(cirq.PauliRotation(sparse).pauli_string, cirq.PauliString)
    assert cirq.PauliRotation(sparse).pauli_string == sparse
    assert isinstance(cirq.PauliRotation(dense, qubits=[q0, q1, q2]).pauli_string, cirq.DensePauliString)
    assert cirq.PauliRotation(dense, qubits=[q0, q1, q2]).pauli_string == dense
    assert isinstance(
        cirq.PauliRotation(dense, 0.5, [q0, q1, q2]).with_qubits(q1, q2, q3).pauli_string, cirq.DensePauliString
    )
    assert isinstance(
        cirq.resolve_parameters(
            cirq.PauliRotation(dense, sympy.Symbol('t'), qubits=[q0, q1, q2]), {'t': 1}
        ).pauli_string,
        cirq.DensePauliString,
    )
    assert isinstance((cirq.PauliRotation(dense, 0.5, [q0, q1, q2]) ** 2).pauli_string, cirq.DensePauliString)


def test_qubits() -> None:
    # PauliString drops identities, DensePauliString keeps them.
    assert cirq.PauliRotation(cirq.X(q0) * cirq.I(q1) * cirq.Z(q2)).qubits == (q0, q2)
    assert cirq.PauliRotation(cirq.DensePauliString('XIZ'), qubits=[q3, q2, q1]).qubits == (
        q3,
        q2,
        q1,
    )


def test_with_qubits() -> None:
    rot = cirq.PauliRotation(cirq.X(q0) * cirq.Y(q1), 0.3)
    assert rot.with_qubits(q2, q3) == cirq.PauliRotation(cirq.X(q2) * cirq.Y(q3), 0.3)
    dense_rot = cirq.PauliRotation(cirq.DensePauliString('XIY'), 0.3, qubits=[q0, q1, q2])
    assert dense_rot.with_qubits(q1, q2, q3) == cirq.PauliRotation(
        cirq.DensePauliString('XIY'), 0.3, qubits=[q1, q2, q3]
    )


@pytest.mark.parametrize(
    'pauli, k',
    (
        (cirq.X(q0), np.pi / 2),
        (cirq.X(q0) * cirq.Y(q1), 0.3),
        (-cirq.Z(q0) * cirq.X(q2), 1),
        (2 * cirq.X(q0) * cirq.Z(q1), 0.7),
    ),
)
def test_unitary_pauli_strings(pauli, k) -> None:
    rot = cirq.PauliRotation(pauli, k)
    assert cirq.has_unitary(rot)
    np.testing.assert_allclose(cirq.unitary(rot), _expected_unitary(pauli, k), atol=1e-8)
    cirq.testing.assert_decompose_is_consistent_with_unitary(rot)


@pytest.mark.parametrize(
    'pauli, k, qubits',
    (
        (cirq.DensePauliString('IX'), np.pi / 4, [q0, q1]),
        (cirq.DensePauliString('XI'), np.pi / 4, [q0, q1]),
        (cirq.DensePauliString('XIX'), np.pi / 4, [q0, q1, q2]),
        (cirq.DensePauliString('IXI'), np.pi / 4, [q0, q1, q2]),
        (cirq.DensePauliString('XIZ'), 0.4, [q0, q1, q2]),
        (cirq.DensePauliString('YX', coefficient=-1), 1.2, [q0, q1]),
        (cirq.DensePauliString('II', coefficient=3), 0.2, [q0, q1]),
    ),
)
def test_unitary_dense_pauli_strings(pauli, k, qubits) -> None:
    rot = cirq.PauliRotation(pauli, k, qubits)
    assert cirq.has_unitary(rot)
    np.testing.assert_allclose(cirq.unitary(rot), _expected_unitary(pauli, k), atol=1e-8)
    cirq.testing.assert_decompose_is_consistent_with_unitary(rot)


def test_parameterized() -> None:
    t = sympy.Symbol('t')
    rot = cirq.PauliRotation(cirq.X(q0) * cirq.Z(q1), t)
    assert cirq.is_parameterized(rot)
    assert cirq.parameter_names(rot) == {'t'}
    assert not cirq.has_unitary(rot)
    with pytest.raises(ValueError, match='parameterized'):
        rot.matrix()
    resolved = cirq.resolve_parameters(rot, {'t': 0.5})
    assert resolved == cirq.PauliRotation(cirq.X(q0) * cirq.Z(q1), 0.5)
    assert not cirq.is_parameterized(resolved)


def test_pow() -> None:
    rot = cirq.PauliRotation(cirq.X(q0) * cirq.Y(q1), 0.3)
    assert rot**2 == cirq.PauliRotation(cirq.X(q0) * cirq.Y(q1), 0.6)
    np.testing.assert_allclose(cirq.unitary(rot**2), cirq.unitary(rot) @ cirq.unitary(rot))
    dense_rot = cirq.PauliRotation(cirq.DensePauliString('XY'), 0.3, qubits=[q2, q3])
    assert dense_rot**-1 == cirq.PauliRotation(cirq.DensePauliString('XY'), -0.3, qubits=[q2, q3])


@pytest.mark.parametrize(
    'rot',
    (
        cirq.PauliRotation(cirq.X(q0) * cirq.Y(q1), 0.3),
        cirq.PauliRotation(cirq.X(q0), sympy.Symbol('t')),
        cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.3, qubits=[q1, q2, q3]),
        cirq.PauliRotation(cirq.DensePauliString('XZ', coefficient=-1), 0.3, qubits=[q2, q3]),
    ),
)
def test_repr(rot) -> None:
    cirq.testing.assert_equivalent_repr(rot)


def test_str() -> None:
    assert (
        str(cirq.PauliRotation(cirq.X(q0) * cirq.Y(q1), 0.5)) == 'exp(j * 0.5 * (X(q(0))*Y(q(1))))'
    )
    assert (
        str(cirq.PauliRotation(cirq.DensePauliString('XIZ'), 0.5, [q0, q1, q2]))
        == 'exp(j * 0.5 * (+XIZ)) on q(0), q(1), q(2)'
    )
