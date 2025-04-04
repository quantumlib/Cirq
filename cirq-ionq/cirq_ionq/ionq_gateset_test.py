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
import sympy

import cirq
import cirq_ionq as ionq

VALID_GATES = (
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.X**0.5,
    cirq.Y**0.5,
    cirq.Z**0.5,
    cirq.rx(0.1),
    cirq.ry(0.1),
    cirq.rz(0.1),
    cirq.H,
    cirq.HPowGate(exponent=1, global_shift=-0.5),
    cirq.T,
    cirq.S,
    cirq.CNOT,
    cirq.CXPowGate(exponent=1, global_shift=-0.5),
    cirq.XX,
    cirq.YY,
    cirq.ZZ,
    cirq.XX**0.5,
    cirq.YY**0.5,
    cirq.ZZ**0.5,
    cirq.SWAP,
    cirq.SwapPowGate(exponent=1, global_shift=-0.5),
    cirq.MeasurementGate(num_qubits=1, key='a'),
    cirq.MeasurementGate(num_qubits=2, key='b'),
    cirq.MeasurementGate(num_qubits=10, key='c'),
)

ionq_target_gateset = ionq.IonQTargetGateset()


@pytest.mark.parametrize('g', [ionq.IonQTargetGateset(), ionq.IonQTargetGateset(atol=1e-5)])
def test_gateset_repr(g):
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ionq\n')


@pytest.mark.parametrize('gate', VALID_GATES)
def test_decompose_leaves_supported_alone(gate):
    qubits = cirq.LineQubit.range(gate.num_qubits())
    operation = gate(*qubits)
    assert gate in ionq_target_gateset
    assert operation in ionq_target_gateset
    circuit = cirq.Circuit(operation)
    decomposed_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=ionq_target_gateset, ignore_failures=False
    )
    assert decomposed_circuit == circuit


VALID_DECOMPOSED_GATES = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CNOT)


def test_decompose_single_qubit_matrix_gate():
    q = cirq.LineQubit(0)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2))
        circuit = cirq.Circuit(gate(q))
        decomposed_circuit = cirq.optimize_for_target_gateset(
            circuit, gateset=ionq_target_gateset, ignore_failures=False
        )
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


def test_decompose_two_qubit_matrix_gate():
    q0, q1 = cirq.LineQubit.range(2)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.optimize_for_target_gateset(
            circuit, gateset=ionq_target_gateset, ignore_failures=False
        )
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, decomposed_circuit, atol=1e-8
        )
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)


@pytest.mark.parametrize(
    'gate, qubits',
    [
        (cirq.CCZ, 3),
        (cirq.QuantumFourierTransformGate(6), 6),
        (cirq.MatrixGate(cirq.testing.random_unitary(8)), 3),
    ],
)
def test_decompose_multi_qubit_cirq_gates(gate, qubits):
    circuit = cirq.Circuit(gate(*cirq.LineQubit.range(qubits)))
    decomposed_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=ionq_target_gateset, ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, decomposed_circuit, atol=1e-8
    )
    assert ionq_target_gateset.validate(decomposed_circuit)


def test_decompose_parameterized_operation():
    op = cirq.ISWAP(*cirq.LineQubit.range(2))
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(op**theta)
    decomposed_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=ionq_target_gateset, ignore_failures=False
    )
    for theta_val in [-0.25, 1.0, 0.5]:
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.resolve_parameters(circuit, {theta: theta_val}),
            cirq.resolve_parameters(decomposed_circuit, {theta: theta_val}),
            atol=1e-6,
        )
    assert ionq_target_gateset.validate(decomposed_circuit)


def test_decomposition_all_to_all_connectivity():
    """This function only accepts 3 qubits as input"""
    with pytest.raises(ValueError):
        decompose_result = ionq.decompose_all_to_all_connect_ccz_gate(
            cirq.CCZ, cirq.LineQubit.range(4)
        )

    decompose_result = ionq.decompose_all_to_all_connect_ccz_gate(cirq.CCZ, cirq.LineQubit.range(3))

    cirq.testing.assert_has_diagram(
        cirq.Circuit(decompose_result),
        """
0: ──────────────@──────────────────@───@───T──────@───
                 │                  │   │          │
1: ───@──────────┼───────@───T──────┼───X───T^-1───X───
      │          │       │          │
2: ───X───T^-1───X───T───X───T^-1───X───T──────────────
""",
    )


def test_decompose_toffoli_gate():
    """Decompose result should reflect all-to-all connectivity"""
    circuit = cirq.Circuit(cirq.TOFFOLI(*cirq.LineQubit.range(3)))
    decomposed_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=ionq_target_gateset, ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, decomposed_circuit, atol=1e-8
    )
    assert ionq_target_gateset.validate(decomposed_circuit)
    cirq.testing.assert_has_diagram(
        decomposed_circuit,
        """
0: ──────────────────@──────────────────@───@───T──────@───
                     │                  │   │          │
1: ───────@──────────┼───────@───T──────┼───X───T^-1───X───
          │          │       │          │
2: ───H───X───T^-1───X───T───X───T^-1───X───T───H──────────
""",
    )
