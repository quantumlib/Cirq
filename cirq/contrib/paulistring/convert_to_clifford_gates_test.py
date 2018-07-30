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

import cirq
from cirq.contrib.paulistring import (
    ConvertToCliffordGates,
)


def test_convert():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        cirq.X(q0),
        cirq.Y(q1) ** 0.5,
        cirq.Z(q0) ** -0.5,
        cirq.Z(q1) ** 0,
        cirq.H(q0),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToCliffordGates().optimize_circuit(circuit)

    assert all(isinstance(op.gate, cirq.CliffordGate)
               for op in circuit.all_operations())

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.to_unitary_matrix(),
        c_orig.to_unitary_matrix(),
        atol=1e-7)

    assert circuit.to_text_diagram() == """
0: ───X───────Z^-0.5───H───

1: ───Y^0.5───I────────────
""".strip()


def test_non_clifford_known_matrix():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(
        cirq.Z(q0) ** 0.25,
    )
    c_orig = cirq.Circuit(circuit)

    ConvertToCliffordGates(ignore_failures=True).optimize_circuit(circuit)
    assert circuit == c_orig

    circuit2 = cirq.Circuit(c_orig)
    with pytest.raises(ValueError):
        ConvertToCliffordGates().optimize_circuit(circuit2)



def test_already_converted():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(
        cirq.CliffordGate.H(q0),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToCliffordGates().optimize_circuit(circuit)

    assert circuit == c_orig


def test_convert_composite():
    class CompositeDummy(cirq.Gate, cirq.CompositeGate):
        def default_decompose(self, qubits):
            q0, q1 = qubits
            yield cirq.X(q0)
            yield cirq.Y(q1) ** 0.5
            yield cirq.H(q0)

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        CompositeDummy()(q0, q1)
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToCliffordGates().optimize_circuit(circuit)

    assert all(isinstance(op.gate, cirq.CliffordGate)
               for op in circuit.all_operations())

    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.to_unitary_matrix(),
        c_orig.to_unitary_matrix(),
        atol=1e-7)

    assert circuit.to_text_diagram() == """
0: ───X───────H───

1: ───Y^0.5───────
""".strip()


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.Gate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        UnsupportedDummy()(q0, q1),
    )
    c_orig = cirq.Circuit(circuit)
    ConvertToCliffordGates(ignore_failures=True).optimize_circuit(circuit)

    assert circuit == c_orig


def test_fail_unsupported_gate():
    class UnsupportedDummy(cirq.Gate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(
        UnsupportedDummy()(q0, q1),
    )
    with pytest.raises(TypeError):
        ConvertToCliffordGates().optimize_circuit(circuit)


def test_rotation_to_clifford_gate():
    conv = ConvertToCliffordGates()

    assert (conv._rotation_to_clifford_gate(cirq.Pauli.X, 0.0)
            == cirq.CliffordGate.I)
    assert (conv._rotation_to_clifford_gate(cirq.Pauli.X, 0.5)
            == cirq.CliffordGate.X_sqrt)
    assert (conv._rotation_to_clifford_gate(cirq.Pauli.X, 1.0)
            == cirq.CliffordGate.X)
    assert (conv._rotation_to_clifford_gate(cirq.Pauli.X, -0.5)
            == cirq.CliffordGate.X_nsqrt)
