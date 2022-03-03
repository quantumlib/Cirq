# Copyright 2022 The Cirq Developers
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
import cirq
import sympy
import numpy as np


def test_instantiate():
    gset = cirq.ApproximateTwoQubitTargetGateset(cirq.CZ)
    assert gset.base_gate == cirq.CZ
    assert cirq.CZ in gset
    assert cirq.H in gset
    assert np.all(gset.tabulation.base_gate == cirq.unitary(cirq.CZ))

    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))
    c = cirq.optimize_for_target_gateset(c, gateset=gset)
    assert (
        len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1
    ), 'It should take 1 CZ gates to decompose a CX gate'


def test_bad_instantiate():
    with pytest.raises(ValueError, match="1"):
        _ = cirq.ApproximateTwoQubitTargetGateset(cirq.H)


def test_correctness():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.H(b),
        cirq.SWAP(a, b) ** 0.5,
        cirq.Y(a) ** 0.456,
        cirq.Y(b) ** 0.123,
        cirq.CNOT(a, b),
        cirq.X(a) ** 0.123,
        cirq.Y(b) ** 0.9,
        cirq.CNOT(b, a),
    )
    c_new = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ, random_state=123)
    )
    print(circuit.final_state_vector())
    assert len(c_new) == 7  # only need 3 CZs.
    assert cirq.fidelity(c_new.final_state_vector(), circuit.final_state_vector()) > 0.995


def test_optimizes_same_gate():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))
    c2 = cirq.optimize_for_target_gateset(
        c, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.ISWAP)
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c, c2, atol=1e-6)

    c = cirq.Circuit(cirq.CX(a, b) ** 0.5)
    c2 = cirq.optimize_for_target_gateset(
        c, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CX ** 0.5)
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c, c2, atol=1e-6)


def test_optimizes_tagged_gate():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit((cirq.CZ ** 0.5)(a, b).with_tags('mytag'))
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ, random_state=123)
    )
    assert (
        len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2
    ), 'It should take 2 CZ gates to decompose a CZ**0.5 gate'


def test_symbols_not_supported():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit((cirq.CZ ** sympy.Symbol('oops'))(a, b))
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ, random_state=123)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_avoids_decompose_when_matrix_available():
    class OtherXX(cirq.testing.TwoQubitGate):
        # coverage: ignore
        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    class OtherOtherXX(cirq.testing.TwoQubitGate):
        # coverage: ignore
        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(OtherXX()(a, b), OtherOtherXX()(a, b))
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ))
    assert len(c) == 0


def test_composite_gates_without_matrix():
    class CompositeDummy(cirq.SingleQubitGate):
        def _decompose_(self, qubits):
            yield cirq.X(qubits[0])
            yield cirq.Y(qubits[0]) ** 0.5

    class CompositeDummy2(cirq.testing.TwoQubitGate):
        def _decompose_(self, qubits):
            yield cirq.CZ(qubits[0], qubits[1])
            yield CompositeDummy()(qubits[1])

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        CompositeDummy()(q0),
        CompositeDummy2()(q0, q1),
    )
    expected = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q0) ** 0.5,
        cirq.CZ(q0, q1),
        cirq.X(q1),
        cirq.Y(q1) ** 0.5,
    )
    c_new = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ, random_state=123)
    )

    assert len(c_new) == 3
    assert cirq.fidelity(c_new.final_state_vector(), expected.final_state_vector()) > 0.995


def test_unsupported_gate():
    class UnsupportedDummy(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnsupportedDummy()(q0, q1))
    assert circuit == cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ)
    )
    with pytest.raises(ValueError, match='Unable to convert'):
        _ = cirq.optimize_for_target_gateset(
            circuit, gateset=cirq.ApproximateTwoQubitTargetGateset(cirq.CZ), ignore_failures=False
        )
