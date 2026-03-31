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

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

import cirq


class Fixed(cirq.Operation):
    def __init__(self, unitary: np.ndarray, qasm: str) -> None:
        self.unitary = unitary
        self.qasm = qasm

    def _unitary_(self):
        return self.unitary

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return tuple(cirq.LineQubit.range(self.unitary.shape[0].bit_length() - 1))

    def with_qubits(self, *new_qubits) -> Fixed:
        raise NotImplementedError()

    def _qasm_(self, args: cirq.QasmArgs):
        return args.format(self.qasm, *self.qubits)


class QuditGate(cirq.Gate):
    def _qid_shape_(self) -> tuple[int, ...]:
        return (3, 3)

    def _unitary_(self):
        return np.eye(9)

    def _qasm_(self, args: cirq.QasmArgs, qubits: tuple[cirq.Qid, ...]):
        return NotImplemented


def test_assert_qasm_is_consistent_with_unitary() -> None:
    if importlib.util.find_spec('qiskit') is None:  # pragma: no cover
        warnings.warn(
            "Skipped test_assert_qasm_is_consistent_with_unitary "
            "because qiskit isn't installed to verify against."
        )
        return

    # Checks matrix.
    cirq.testing.assert_qasm_is_consistent_with_unitary(
        Fixed(np.array([[1, 0], [0, 1]]), 'z {0}; z {0};')
    )
    cirq.testing.assert_qasm_is_consistent_with_unitary(
        Fixed(np.array([[1, 0], [0, -1]]), 'z {0};')
    )
    with pytest.raises(AssertionError, match='Not equal'):
        cirq.testing.assert_qasm_is_consistent_with_unitary(
            Fixed(np.array([[1, 0], [0, -1]]), 'x {0};')
        )

    # Checks qubit ordering.
    cirq.testing.assert_qasm_is_consistent_with_unitary(cirq.CNOT)
    cirq.testing.assert_qasm_is_consistent_with_unitary(
        cirq.CNOT.on(cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    )
    cirq.testing.assert_qasm_is_consistent_with_unitary(
        cirq.CNOT.on(cirq.NamedQubit('b'), cirq.NamedQubit('a'))
    )

    # Checks that code is valid.
    with pytest.raises(AssertionError, match='QASM not consistent'):
        cirq.testing.assert_qasm_is_consistent_with_unitary(
            Fixed(np.array([[1, 0], [0, -1]]), 'JUNK$&*@($#::=[];')
        )

    # Checks that the test handles qudits
    cirq.testing.assert_qasm_is_consistent_with_unitary(QuditGate())
