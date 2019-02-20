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

import numpy as np
import pytest

import cirq


class NoMethod:
    pass


class ReturnsNotImplemented:
    def _pauli_expansion_(self):
        return NotImplemented


class ReturnsCoefficients:
    def __init__(self, coefficients: np.ndarray):
        self._coefficients = coefficients

    def _pauli_expansion_(self) -> np.ndarray:
        return self._coefficients


class HasUnitary:
    def __init__(self, unitary: np.ndarray):
        self._unitary = unitary

    def _unitary_(self) -> np.ndarray:
        return self._unitary


@pytest.mark.parametrize('val', (
    NoMethod(),
    ReturnsNotImplemented(),
    123,
    np.eye(2),
    object(),
    cirq,
))
def test_no_pauli_expansion(val):
    assert cirq.pauli_expansion(val) is NotImplemented


@pytest.mark.parametrize('val, expected_coefficients', (
    (ReturnsCoefficients(np.array([1, 2, 3, 4])), np.array([1, 2, 3, 4])),
    (HasUnitary(np.eye(2)), np.array([1, 0, 0, 0])),
    (HasUnitary(np.array([[1, -1j], [1j, -1]])), np.array([0, 0, 1, 1])),
    (HasUnitary(np.array([[0., 1.], [0., 0.]])), np.array([0, 0.5, 0.5j, 0])),
    (cirq.H, np.array([0, 1, 0, 1]) / np.sqrt(2)),
    (cirq.Ry(np.pi / 2), np.array([1, 0, -1j, 0]) / np.sqrt(2)),
))
def test_pauli_expansion(val, expected_coefficients):
    assert np.allclose(cirq.pauli_expansion(val), expected_coefficients,
                       rtol=0, atol=1e-12)
