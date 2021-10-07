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
"""Tests for channels."""
from typing import Iterable, Sequence

import numpy as np
import pytest

import cirq


def apply_channel(channel: cirq.SupportsKraus, rho: np.ndarray) -> np.ndarray:
    return apply_kraus_operators(cirq.kraus(channel), rho)


def apply_kraus_operators(kraus_operators: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    d_out, d_in = kraus_operators[0].shape
    assert rho.shape == (d_in, d_in)
    out = np.zeros((d_out, d_out), dtype=np.complex128)
    for k in kraus_operators:
        out += k @ rho @ k.conj().T
    return out


def generate_standard_operator_basis(d_out: int, d_in: int) -> Iterable[np.ndarray]:
    for i in range(d_out):
        for j in range(d_in):
            e_ij = np.zeros((d_out, d_in))
            e_ij[i, j] = 1
            yield e_ij


def compute_choi(channel: cirq.SupportsKraus) -> np.ndarray:
    ks = cirq.kraus(channel)
    d_out, d_in = ks[0].shape
    d = d_in * d_out
    c = np.zeros((d, d), dtype=np.complex128)
    for e in generate_standard_operator_basis(d_in, d_in):
        c += np.kron(apply_channel(channel, e), e)
    return c


def compute_superoperator(channel: cirq.SupportsKraus) -> np.ndarray:
    ks = cirq.kraus(channel)
    d_out, d_in = ks[0].shape
    m = np.zeros((d_out * d_out, d_in * d_in), dtype=np.complex128)
    for k, e_in in enumerate(generate_standard_operator_basis(d_in, d_in)):
        m[:, k] = np.reshape(apply_channel(channel, e_in), d_out * d_out)
    return m


@pytest.mark.parametrize(
    'kraus_operators, expected_choi',
    (
        ([np.eye(2)], np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])),
        (cirq.kraus(cirq.depolarize(0.75)), np.eye(4) / 2),
        (
            [
                np.array([[1, 0, 0], [0, 0, 1]]) / np.sqrt(2),
                np.array([[1, 0, 0], [0, 0, -1]]) / np.sqrt(2),
            ],
            np.diag([1, 0, 0, 0, 0, 1]),
        ),
    ),
)
def test_kraus_to_choi(kraus_operators, expected_choi):
    """Verifies that cirq.kraus_to_choi computes the correct Choi matrix."""
    assert np.allclose(cirq.kraus_to_choi(kraus_operators), expected_choi)


@pytest.mark.parametrize(
    'choi, error',
    (
        (np.array([[1, 2, 3], [4, 5, 6]]), "shape"),
        (np.eye(2), "shape"),
        (np.diag([1, 1, 1, -1]), "positive"),
        (
            np.array(
                [
                    [0.6, 0.0, -0.1j, 0.1],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1j, 0.0, 0.4, 0.0],
                    [0.2, 0.0, 0.0, 1.0],
                ]
            ),
            "Hermitian",
        ),
    ),
)
def test_choi_to_kraus_invalid_input(choi, error):
    with pytest.raises(ValueError, match=error):
        _ = cirq.choi_to_kraus(choi)


@pytest.mark.parametrize(
    'choi, expected_kraus',
    (
        (
            # Identity channel
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            (np.eye(2),),
        ),
        (
            # S gate
            np.array([[1, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 1]]),
            (np.diag([-1j, 1]),),
        ),
        (
            # Hadamard
            np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2,
            (np.array([[1, 1], [1, -1]]) / np.sqrt(2),),
        ),
        (
            # Completely dephasing channel
            np.diag([1, 0, 0, 1]),
            (np.diag([1, 0]), np.diag([0, 1])),
        ),
        (
            # Amplitude damping channel
            np.array(
                [
                    [1, 0, 0, 0.8],
                    [0, 0.36, 0, 0],
                    [0, 0, 0, 0],
                    [0.8, 0, 0, 0.64],
                ],
            ),
            (np.diag([1, 0.8]), np.array([[0, 0.6], [0, 0]])),
        ),
        (
            # Completely depolarizing channel
            np.eye(4) / 2,
            (
                np.array([[np.sqrt(0.5), 0], [0, 0]]),
                np.array([[0, np.sqrt(0.5)], [0, 0]]),
                np.array([[0, 0], [np.sqrt(0.5), 0]]),
                np.array([[0, 0], [0, np.sqrt(0.5)]]),
            ),
        ),
    ),
)
def test_choi_to_kraus_fixed_values(choi, expected_kraus):
    """Verifies that cirq.choi_to_kraus gives correct results on a few fixed inputs."""
    actual_kraus = cirq.choi_to_kraus(choi)
    assert len(actual_kraus) == len(expected_kraus)
    for i in (0, 1):
        for j in (0, 1):
            input_rho = np.zeros((2, 2))
            input_rho[i, j] = 1
            actual_rho = apply_kraus_operators(actual_kraus, input_rho)
            expected_rho = apply_kraus_operators(expected_kraus, input_rho)
            assert np.allclose(actual_rho, expected_rho)


@pytest.mark.parametrize(
    'choi',
    (
        # Identity channel
        np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
        # Unitary bit flip channel
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
        # Constant channel
        np.eye(4) / 2,
        # Completely dephasing channel
        np.diag([1, 0, 0, 1]),
        # A channel with mixed output on computational basis
        np.array(
            [
                [0.6, 0.0, -0.1j, 0.1],
                [0.0, 0.1, 0.0, 0.1j],
                [0.1j, 0.0, 0.4, 0.0],
                [0.1, -0.1j, 0.0, 0.9],
            ]
        ),
    ),
)
def test_choi_to_kraus_action_on_operatorial_basis(choi):
    """Verifies that cirq.choi_to_kraus computes a valid Kraus representation."""
    kraus_operators = cirq.choi_to_kraus(choi)
    c = np.reshape(choi, (2, 2, 2, 2))
    for i in (0, 1):
        for j in (0, 1):
            input_rho = np.zeros((2, 2))
            input_rho[i, j] = 1
            actual_rho = apply_kraus_operators(kraus_operators, input_rho)
            expected_rho = c[:, i, :, j]
            assert np.allclose(actual_rho, expected_rho)


@pytest.mark.parametrize(
    'choi',
    (
        np.eye(4),
        np.diag([1, 0, 0, 1]),
        np.diag([0.2, 0.3, 0.8, 0.7]),
        np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, -1],
                [1, 0, 1, 0],
                [0, -1, 0, 1],
            ]
        ),
        np.array(
            [
                [0.8, 0, 0, 0.5],
                [0, 0.3, 0, 0],
                [0, 0, 0.2, 0],
                [0.5, 0, 0, 0.7],
            ],
        ),
    ),
)
def test_choi_to_kraus_inverse_of_kraus_to_choi(choi):
    """Verifies that cirq.kraus_to_choi(cirq.choi_to_kraus(.)) is identity on Choi matrices."""
    kraus = cirq.choi_to_kraus(choi)
    recovered_choi = cirq.kraus_to_choi(kraus)
    assert np.allclose(recovered_choi, choi)


@pytest.mark.parametrize(
    'channel',
    (
        cirq.I,
        cirq.X,
        cirq.CNOT,
        cirq.depolarize(0.1),
        cirq.depolarize(0.1, n_qubits=2),
        cirq.amplitude_damp(0.2),
    ),
)
def test_operation_to_choi(channel):
    """Verifies that cirq.operation_to_choi correctly computes the Choi matrix."""
    n_qubits = cirq.num_qubits(channel)
    actual = cirq.operation_to_choi(channel)
    expected = compute_choi(channel)
    assert np.isclose(np.trace(actual), 2 ** n_qubits)
    assert np.all(actual == expected)


def test_choi_for_completely_dephasing_channel():
    """Checks cirq.operation_to_choi on the completely dephasing channel."""
    assert np.all(cirq.operation_to_choi(cirq.phase_damp(1)) == np.diag([1, 0, 0, 1]))


@pytest.mark.parametrize(
    'kraus_operators, expected_superoperator',
    (
        ([np.eye(2)], np.eye(4)),
        (
            cirq.kraus(cirq.depolarize(0.75)),
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2,
        ),
        (
            [
                np.array([[0, 1, 0], [0, 0, 1]]) / np.sqrt(2),
                np.array([[0, 1, 0], [0, 0, -1]]) / np.sqrt(2),
            ],
            np.array(
                [
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        ),
    ),
)
def test_kraus_to_superoperator(kraus_operators, expected_superoperator):
    """Verifies that cirq.kraus_to_superoperator computes the correct channel matrix."""
    assert np.allclose(cirq.kraus_to_superoperator(kraus_operators), expected_superoperator)
    with cirq.testing.assert_deprecated(deadline='v0.14'):
        assert np.allclose(cirq.kraus_to_channel_matrix(kraus_operators), expected_superoperator)


@pytest.mark.parametrize(
    'superoperator, expected_kraus_operators',
    (
        (np.eye(4), [np.eye(2)]),
        (np.diag([1, -1, -1, 1]), [np.diag([1, -1])]),
        (np.diag([1, -1j, 1j, 1]), [np.diag([1, 1j])]),
        (
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2,
            cirq.kraus(cirq.depolarize(0.75)),
        ),
    ),
)
def test_superoperator_to_kraus_fixed_values(superoperator, expected_kraus_operators):
    """Verifies that cirq.kraus_to_superoperator computes the correct channel matrix."""
    actual_kraus_operators = cirq.superoperator_to_kraus(superoperator)
    for i in (0, 1):
        for j in (0, 1):
            input_rho = np.zeros((2, 2))
            input_rho[i, j] = 1
            actual_rho = apply_kraus_operators(actual_kraus_operators, input_rho)
            expected_rho = apply_kraus_operators(expected_kraus_operators, input_rho)
            assert np.allclose(actual_rho, expected_rho)


@pytest.mark.parametrize(
    'superoperator',
    (
        np.eye(4),
        np.diag([1, 0, 0, 1]),
        np.diag([1, -1j, 1j, 1]),
        np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [1, 0, 0, 0.8],
                [0, 0.36, 0, 0],
                [0, 0, 0.36, 0],
                [0, 0, 0, 0.64],
            ],
        ),
    ),
)
def test_superoperator_to_kraus_inverse_of_kraus_to_superoperator(superoperator):
    """Verifies that cirq.kraus_to_superoperator(cirq.superoperator_to_kraus(.)) is identity."""
    kraus = cirq.superoperator_to_kraus(superoperator)
    recovered_superoperator = cirq.kraus_to_superoperator(kraus)
    assert np.allclose(recovered_superoperator, superoperator)


@pytest.mark.parametrize(
    'choi, error',
    (
        (np.array([[1, 2, 3], [4, 5, 6]]), "shape"),
        (np.eye(2), "shape"),
        (
            np.array(
                [
                    [0.6, 0.0, -0.1j, 0.1],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1j, 0.0, 0.4, 0.0],
                    [0.2, 0.0, 0.0, 1.0],
                ]
            ),
            "Hermitian",
        ),
    ),
)
def test_choi_to_superoperator_invalid_input(choi, error):
    with pytest.raises(ValueError, match=error):
        _ = cirq.choi_to_superoperator(choi)


@pytest.mark.parametrize(
    'superoperator, error',
    (
        (np.array([[1, 2, 3], [4, 5, 6]]), "shape"),
        (np.eye(2), "shape"),
    ),
)
def test_superoperator_to_choi_invalid_input(superoperator, error):
    with pytest.raises(ValueError, match=error):
        _ = cirq.superoperator_to_choi(superoperator)


@pytest.mark.parametrize(
    'superoperator, choi',
    (
        (
            # Identity channel
            np.eye(4),
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
        ),
        (
            # S gate
            np.diag([1, -1j, 1j, 1]),
            np.array([[1, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 1]]),
        ),
        (
            # Hadamard
            np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2,
            np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2,
        ),
        (
            # Completely dephasing channel
            np.diag([1, 0, 0, 1]),
            np.diag([1, 0, 0, 1]),
        ),
        (
            # Amplitude damping channel
            np.array(
                [
                    [1, 0, 0, 0.36],
                    [0, 0.8, 0, 0],
                    [0, 0, 0.8, 0],
                    [0, 0, 0, 0.64],
                ],
            ),
            np.array(
                [
                    [1, 0, 0, 0.8],
                    [0, 0.36, 0, 0],
                    [0, 0, 0, 0],
                    [0.8, 0, 0, 0.64],
                ],
            ),
        ),
        (
            # Completely depolarizing channel
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2,
            np.eye(4) / 2,
        ),
    ),
)
def test_superoperator_vs_choi_fixed_values(superoperator, choi):
    recovered_choi = cirq.superoperator_to_choi(superoperator)
    assert np.allclose(recovered_choi, choi)

    recovered_superoperator = cirq.choi_to_superoperator(choi)
    assert np.allclose(recovered_superoperator, superoperator)


@pytest.mark.parametrize(
    'choi',
    (
        np.eye(4),
        np.diag([1, 0, 0, 1]),
        np.diag([0.2, 0.3, 0.8, 0.7]),
        np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, -1],
                [1, 0, 1, 0],
                [0, -1, 0, 1],
            ]
        ),
        np.array(
            [
                [0.8, 0, 0, 0.5],
                [0, 0.3, 0, 0],
                [0, 0, 0.2, 0],
                [0.5, 0, 0, 0.7],
            ],
        ),
    ),
)
def test_choi_to_superoperator_inverse_of_superoperator_to_choi(choi):
    superoperator = cirq.choi_to_superoperator(choi)
    recovered_choi = cirq.superoperator_to_choi(superoperator)
    assert np.allclose(recovered_choi, choi)

    recovered_superoperator = cirq.choi_to_superoperator(recovered_choi)
    assert np.allclose(recovered_superoperator, superoperator)


@pytest.mark.parametrize(
    'channel',
    (
        cirq.I,
        cirq.X,
        cirq.CNOT,
        cirq.depolarize(0.1),
        cirq.depolarize(0.1, n_qubits=2),
        cirq.amplitude_damp(0.2),
    ),
)
def test_operation_to_superoperator(channel):
    """Verifies that cirq.operation_to_superoperator correctly computes the channel matrix."""
    expected = compute_superoperator(channel)
    assert np.all(expected == cirq.operation_to_superoperator(channel))
    with cirq.testing.assert_deprecated(deadline='v0.14'):
        assert np.all(expected == cirq.operation_to_channel_matrix(channel))


def test_superoperator_for_completely_dephasing_channel():
    """Checks cirq.operation_to_superoperator on the completely dephasing channel."""
    assert np.all(cirq.operation_to_superoperator(cirq.phase_damp(1)) == np.diag([1, 0, 0, 1]))
