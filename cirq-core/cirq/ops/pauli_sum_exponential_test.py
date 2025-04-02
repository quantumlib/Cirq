# Copyright 2021 The Cirq Developers
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
import cirq.testing

q0, q1, q2, q3 = cirq.LineQubit.range(4)


def test_raises_for_non_commuting_paulis():
    with pytest.raises(ValueError, match='commuting'):
        cirq.PauliSumExponential(cirq.X(q0) + cirq.Z(q0), np.pi / 2)


def test_raises_for_non_hermitian_pauli():
    with pytest.raises(ValueError, match='hermitian'):
        cirq.PauliSumExponential(cirq.X(q0) + 1j * cirq.Z(q1), np.pi / 2)


@pytest.mark.parametrize(
    'psum_exp, expected_qubits',
    (
        (cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2), (q1,)),
        (
            cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol("theta")),
            (q0, q2),
        ),
        (
            cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), np.pi),
            (q0, q1, q2, q3),
        ),
    ),
)
def test_pauli_sum_exponential_qubits(psum_exp, expected_qubits):
    assert psum_exp.qubits == expected_qubits


@pytest.mark.parametrize(
    'psum_exp, expected_psum_exp',
    (
        (
            cirq.PauliSumExponential(cirq.Z(q0), np.pi / 2),
            cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2),
        ),
        (
            cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol("theta")),
            cirq.PauliSumExponential(2j * cirq.X(q1) + 3j * cirq.Y(q3), sympy.Symbol("theta")),
        ),
        (
            cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), np.pi),
            cirq.PauliSumExponential(cirq.X(q1) * cirq.Y(q2) + cirq.Y(q2) * cirq.Z(q3), np.pi),
        ),
    ),
)
def test_pauli_sum_exponential_with_qubits(psum_exp, expected_psum_exp):
    assert psum_exp.with_qubits(*expected_psum_exp.qubits) == expected_psum_exp


@pytest.mark.parametrize(
    'psum, exp',
    (
        (cirq.Z(q0), np.pi / 2),
        (2 * cirq.X(q0) + 3 * cirq.Y(q2), 1),
        (cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), np.pi),
    ),
)
def test_with_parameters_resolved_by(psum, exp):
    psum_exp = cirq.PauliSumExponential(psum, sympy.Symbol("theta"))
    resolver = cirq.ParamResolver({"theta": exp})
    actual = cirq.resolve_parameters(psum_exp, resolver)
    expected = cirq.PauliSumExponential(psum, exp)
    assert actual == expected


def test_pauli_sum_exponential_parameterized_matrix_raises():
    with pytest.raises(ValueError, match='parameterized'):
        cirq.PauliSumExponential(cirq.X(q0) + cirq.Z(q1), sympy.Symbol("theta")).matrix()


@pytest.mark.parametrize(
    'psum_exp, expected_unitary',
    (
        (cirq.PauliSumExponential(cirq.X(q0), np.pi / 2), np.array([[0, 1j], [1j, 0]])),
        (
            cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Z(q1), np.pi / 2),
            np.array([[1j, 0, 0, 0], [0, -1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]]),
        ),
    ),
)
def test_pauli_sum_exponential_has_correct_unitary(psum_exp, expected_unitary):
    assert cirq.has_unitary(psum_exp)
    assert np.allclose(cirq.unitary(psum_exp), expected_unitary)


@pytest.mark.parametrize(
    'psum_exp, power, expected_psum',
    (
        (
            cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2),
            5,
            cirq.PauliSumExponential(cirq.Z(q1), 5 * np.pi / 2),
        ),
        (
            cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol("theta")),
            5,
            cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), 5 * sympy.Symbol("theta")),
        ),
        (
            cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), np.pi),
            5,
            cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), 5 * np.pi),
        ),
    ),
)
def test_pauli_sum_exponential_pow(psum_exp, power, expected_psum):
    assert psum_exp**power == expected_psum


@pytest.mark.parametrize(
    'psum_exp',
    (
        (cirq.PauliSumExponential(0, np.pi / 2)),
        (cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Z(q1), np.pi / 2)),
    ),
)
def test_pauli_sum_exponential_repr(psum_exp):
    cirq.testing.assert_equivalent_repr(psum_exp)


@pytest.mark.parametrize(
    'psum_exp, expected_str',
    (
        (cirq.PauliSumExponential(0, np.pi / 2), 'exp(j * 1.5707963267948966 * (0.000))'),
        (
            cirq.PauliSumExponential(2j * cirq.X(q0) + 4j * cirq.Y(q1), 2),
            'exp(2 * (2.000j*X(q(0))+4.000j*Y(q(1))))',
        ),
        (
            cirq.PauliSumExponential(0.5 * cirq.X(q0) + 0.6 * cirq.Y(q1), sympy.Symbol("theta")),
            'exp(j * theta * (0.500*X(q(0))+0.600*Y(q(1))))',
        ),
    ),
)
def test_pauli_sum_exponential_formatting(psum_exp, expected_str):
    assert str(psum_exp) == expected_str
