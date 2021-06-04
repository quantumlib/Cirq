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
from typing import Iterable

import numpy as np
import pytest

import cirq


def apply_channel(channel: cirq.SupportsChannel, rho: np.ndarray) -> np.ndarray:
    ks = cirq.channel(channel)
    d_out, d_in = ks[0].shape
    assert rho.shape == (d_in, d_in)
    out = np.zeros((d_out, d_out), dtype=np.complex128)
    for k in ks:
        out += k @ rho @ k.conj().T
    return out


def generate_standard_operator_basis(d_out: int, d_in: int) -> Iterable[np.ndarray]:
    for i in range(d_out):
        for j in range(d_in):
            e_ij = np.zeros((d_out, d_in))
            e_ij[i, j] = 1
            yield e_ij


def compute_choi(channel: cirq.SupportsChannel) -> np.ndarray:
    ks = cirq.channel(channel)
    d_out, d_in = ks[0].shape
    d = d_in * d_out
    c = np.zeros((d, d), dtype=np.complex128)
    for e in generate_standard_operator_basis(d_in, d_in):
        c += np.kron(apply_channel(channel, e), e)
    return c


def compute_channel_matrix(channel: cirq.SupportsChannel) -> np.ndarray:
    ks = cirq.channel(channel)
    d_out, d_in = ks[0].shape
    m = np.zeros((d_out * d_out, d_in * d_in), dtype=np.complex128)
    for k, e_in in enumerate(generate_standard_operator_basis(d_in, d_in)):
        m[:, k] = np.reshape(apply_channel(channel, e_in), d_out * d_out)
    return m


@pytest.mark.parametrize(
    'kraus_operators, expected_choi',
    (
        ([np.eye(2)], np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])),
        (cirq.channel(cirq.depolarize(0.75)), np.eye(4) / 2),
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
    'kraus_operators, expected_channel_matrix',
    (
        ([np.eye(2)], np.eye(4)),
        (
            cirq.channel(cirq.depolarize(0.75)),
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
def test_kraus_to_channel_matrix(kraus_operators, expected_channel_matrix):
    """Verifies that cirq.kraus_to_channel_matrix computes the correct channel matrix."""
    assert np.allclose(cirq.kraus_to_channel_matrix(kraus_operators), expected_channel_matrix)


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
def test_operation_to_channel_matrix(channel):
    """Verifies that cirq.channel_matrix correctly computes the channel matrix."""
    actual = cirq.operation_to_channel_matrix(channel)
    expected = compute_channel_matrix(channel)
    assert np.all(actual == expected)


def test_channel_matrix_for_completely_dephasing_channel():
    """Checks cirq.operation_to_channel_matrix on the completely dephasing channel."""
    assert np.all(cirq.operation_to_channel_matrix(cirq.phase_damp(1)) == np.diag([1, 0, 0, 1]))
