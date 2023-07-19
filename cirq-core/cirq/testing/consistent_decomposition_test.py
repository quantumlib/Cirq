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

import pytest

import numpy as np
import sympy

import cirq


class GoodGateDecompose(cirq.testing.SingleQubitGate):
    def _decompose_(self, qubits):
        return cirq.X(qubits[0])

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])


class BadGateDecompose(cirq.testing.SingleQubitGate):
    def _decompose_(self, qubits):
        return cirq.Y(qubits[0])

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])


def test_assert_decompose_is_consistent_with_unitary():
    cirq.testing.assert_decompose_is_consistent_with_unitary(GoodGateDecompose())

    cirq.testing.assert_decompose_is_consistent_with_unitary(
        GoodGateDecompose().on(cirq.NamedQubit('q'))
    )

    cirq.testing.assert_decompose_is_consistent_with_unitary(
        cirq.testing.PhaseUsingCleanAncilla(theta=0.1, ancilla_bitsize=3)
    )

    cirq.testing.assert_decompose_is_consistent_with_unitary(
        cirq.testing.PhaseUsingDirtyAncilla(phase_state=1, ancilla_bitsize=4)
    )

    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_is_consistent_with_unitary(BadGateDecompose())

    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_is_consistent_with_unitary(
            BadGateDecompose().on(cirq.NamedQubit('q'))
        )


class GateDecomposesToDefaultGateset(cirq.Gate):
    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        return [GoodGateDecompose().on(qubits[0]), BadGateDecompose().on(qubits[1])]


class GateDecomposeDoesNotEndInDefaultGateset(cirq.Gate):
    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        yield GateDecomposeNotImplemented().on_each(*qubits)


class GateDecomposeNotImplemented(cirq.testing.SingleQubitGate):
    def _decompose_(self, qubits):
        return NotImplemented


class ParameterizedGate(cirq.Gate):
    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        yield cirq.X(qubits[0]) ** sympy.Symbol("x")
        yield cirq.Y(qubits[1]) ** sympy.Symbol("y")


def test_assert_decompose_ends_at_default_gateset():
    cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposesToDefaultGateset())
    cirq.testing.assert_decompose_ends_at_default_gateset(
        GateDecomposesToDefaultGateset().on(*cirq.LineQubit.range(2))
    )

    cirq.testing.assert_decompose_ends_at_default_gateset(ParameterizedGate())
    cirq.testing.assert_decompose_ends_at_default_gateset(
        ParameterizedGate().on(*cirq.LineQubit.range(2))
    )

    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposeNotImplemented())

    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(
            GateDecomposeNotImplemented().on(cirq.NamedQubit('q'))
        )
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(
            GateDecomposeDoesNotEndInDefaultGateset()
        )

    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(
            GateDecomposeDoesNotEndInDefaultGateset().on(*cirq.LineQubit.range(4))
        )
