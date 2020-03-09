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

import cirq


def test_avoids_decompose_when_matrix_available():

    class OtherXX(cirq.TwoQubitGate):
        # coverage: ignore
        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    class OtherOtherXX(cirq.TwoQubitGate):
        # coverage: ignore
        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(OtherXX()(a, b), OtherOtherXX()(a, b))
    cirq.ConvertToCzAndSingleGates().optimize_circuit(c)
    assert len(c) == 2


def test_kak_decomposes_unknown_two_qubit_gate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.ISWAP(q0, q1))
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates().optimize_circuit(circuit)

    assert sum(1 for op in circuit.all_operations()
                 if len(op.qubits) > 1) == 2
    assert sum(1 for op in circuit.all_operations()
               if isinstance(op.gate, cirq.CZPowGate)) == 2
    assert all(op.gate.exponent == 1
               for op in circuit.all_operations()
               if isinstance(op.gate, cirq.CZPowGate))
    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(),
                                                    c_orig.unitary(),
                                                    atol=1e-7)


def test_composite_gates_without_matrix():
    class CompositeDummy(cirq.SingleQubitGate):
        def _decompose_(self, qubits):
            yield cirq.X(qubits[0])
            yield cirq.Y(qubits[0]) ** 0.5

    class CompositeDummy2(cirq.TwoQubitGate):
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
        cirq.Y(q0)**0.5,
        cirq.CZ(q0, q1),
        cirq.X(q1),
        cirq.Y(q1)**0.5,
    )
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates().optimize_circuit(circuit)

    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(),
                                                    expected.unitary(),
                                                    atol=1e-7)
    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(),
                                                    c_orig.unitary(),
                                                    atol=1e-7)


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnsupportedDummy()(q0, q1),)
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates(ignore_failures=True
                                   ).optimize_circuit(circuit)

    assert circuit == c_orig


def test_fail_unsupported_gate():
    class UnsupportedDummy(cirq.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnsupportedDummy()(q0, q1),)
    with pytest.raises(TypeError):
        cirq.ConvertToCzAndSingleGates().optimize_circuit(circuit)


def test_passes_through_measurements():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='m0'),
        cirq.measure(q1, q2, key='m1', invert_mask=(True, False)),
    )
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates().optimize_circuit(circuit)
    assert circuit == c_orig


def test_allow_partial_czs():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CZ(q0, q1)**0.5,)
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates(allow_partial_czs=True
                                   ).optimize_circuit(circuit)

    assert circuit == c_orig

    # yapf: disable
    circuit2 = cirq.Circuit(
        cirq.MatrixGate((np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1j]]))).on(q0, q1))
    # yapf: enable
    cirq.ConvertToCzAndSingleGates(allow_partial_czs=True
                                   ).optimize_circuit(circuit2)
    two_qubit_ops = list(circuit2.findall_operations(
                                      lambda e: len(e.qubits) == 2))
    assert len(two_qubit_ops) == 1
    gate = two_qubit_ops[0][1].gate
    assert isinstance(gate, cirq.ops.CZPowGate) and gate.exponent == 0.5


def test_dont_allow_partial_czs():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CZ(q0, q1)**0.5,)
    c_orig = cirq.Circuit(circuit)
    cirq.ConvertToCzAndSingleGates(ignore_failures=True
                                   ).optimize_circuit(circuit)

    assert sum(1 for op in circuit.all_operations()
                 if len(op.qubits) > 1) == 2
    assert sum(1 for op in circuit.all_operations()
               if isinstance(op.gate, cirq.CZPowGate)) == 2
    assert all(op.gate.exponent % 2 == 1
               for op in circuit.all_operations()
               if isinstance(op.gate, cirq.CZPowGate))
    cirq.testing.assert_allclose_up_to_global_phase(circuit.unitary(),
                                                    c_orig.unitary(),
                                                    atol=1e-7)
