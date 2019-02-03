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

from typing import Optional

import numpy as np
import pytest

import cirq


class FakeLinearOperator(cirq.AbstractLinearOperator):
    def __init__(
            self,
            matrix: Optional[np.ndarray] = None,
            pauli_expansion: Optional[np.ndarray] = None,
    ) -> None:
        self._matrix = matrix
        self._pauli_expansion = pauli_expansion

    def num_qubits(self) -> int:
        if self._matrix is not None:
            return int(np.log2(self._matrix.shape[0]))
        return 0

    def _matrix_(self) -> Optional[np.ndarray]:
        return self._matrix

    def _pauli_expansion_(self) -> Optional[np.ndarray]:
        return self._pauli_expansion


@pytest.mark.parametrize('op', (
    cirq.LinearOperator(pauli_expansion=np.array([1, 0, 0, 0])),
    cirq.LinearOperator(matrix=np.eye(4)),
    cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                        pauli_expansion=(0, 1, 0, 1)),
    FakeLinearOperator(),
    FakeLinearOperator(np.eye(2)),
    FakeLinearOperator(pauli_expansion=np.array([1, 1, 1, 1])),
))
def test_linear_operator_is_consistent(op):
    cirq.testing.assert_linear_operator_is_consistent(op)


@pytest.mark.parametrize('op', (
    FakeLinearOperator(np.eye(2), np.array([0, 0, 1, 0])),
    FakeLinearOperator(np.diag([1, -1]), np.array([1, 0, 0, 0])),
))
def test_linear_operator_is_inconsistent(op):
    with pytest.raises(AssertionError):
        cirq.testing.assert_linear_operator_is_consistent(op)


@pytest.mark.parametrize('op', (
    cirq.X, cirq.H, cirq.T, cirq.CNOT, cirq.TOFFOLI,
    cirq.LinearOperator(np.array([[1, 2], [3, 4]])),
    cirq.LinearOperator(pauli_expansion=(-1, 0, 1, 2)),
    cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                        pauli_expansion=(0, 1, 0, 1)),
    FakeLinearOperator(np.eye(2), np.array([1, 0, 0, 0])),
))
def test_linear_operator_is_equal_to_itself(op):
    cirq.testing.assert_linear_operators_are_equal(op, op)


@pytest.mark.parametrize('op1, op2', (
    (cirq.X**0, cirq.Y**0),
    (cirq.S, cirq.LinearOperator(np.diag([1, 1j]))),
    (cirq.T, cirq.SingleQubitMatrixGate(np.diag([1, np.sqrt(1j)]))),
    (cirq.CZ, cirq.TwoQubitMatrixGate(np.diag([1, 1, 1, -1]))),
    (cirq.TOFFOLI, cirq.CCXPowGate(exponent=0.5)**2),
    (cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                         pauli_expansion=(0, 1, 0, 1)),
     cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                         pauli_expansion=(0, 1, 0, 1))),
    (cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                         pauli_expansion=(0, 1, 0, 1)),
     cirq.LinearOperator(pauli_expansion=(0, 1, 0, 1))),
    (cirq.LinearOperator(matrix=np.array([[1, 1], [1, -1]]),
                         pauli_expansion=(0, 1, 0, 1)),
     cirq.LinearOperator(np.array([[1, 1], [1, -1]]))),
))
def test_equal_linear_operators_are_equal(op1, op2):
    cirq.testing.assert_linear_operators_are_equal(op1, op2)
    cirq.testing.assert_linear_operators_are_equal(op2, op1)


@pytest.mark.parametrize('op1, op2', (
    (cirq.X, cirq.Y),
    (cirq.X, -cirq.X),
    (cirq.Y, cirq.Y**0.5),
    (cirq.Z, cirq.Rz(np.pi)),
    (cirq.S, cirq.T),
    (cirq.X, 2 * cirq.X),
    (cirq.X, cirq.CNOT),
    (cirq.CNOT, cirq.XX),
    (cirq.SWAP, cirq.ISWAP),
    (cirq.SWAP, cirq.FREDKIN),
    (cirq.FREDKIN, cirq.TOFFOLI),
    (cirq.LinearOperator(np.array([[1, 2], [3, 4]])),
     cirq.LinearOperator(np.array([[1, 2], [3, 4.000000001]]))),
    (cirq.LinearOperator(pauli_expansion=(0, 1, 1, 0)),
     cirq.LinearOperator(pauli_expansion=(0, 1, 1.000000001, 0))),
    (cirq.LinearOperator(matrix=np.array([[1, 1],
                                          [1, -1]]),
                         pauli_expansion=(0, 1, 0, 1)),
     cirq.LinearOperator(matrix=np.array([[1.000000001, 1],
                                          [1, -1.000000001]]),
                         pauli_expansion=(0, 1, 1.000000001, 0))),
))
def test_different_linear_operators_are_not_equal(op1, op2):
    with pytest.raises(AssertionError):
        cirq.testing.assert_linear_operators_are_equal(op1, op2)
    with pytest.raises(AssertionError):
        cirq.testing.assert_linear_operators_are_equal(op2, op1)
