# Copyright 2024 The Cirq Developers
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
"""Tests transpiling of circuits using the IonQ native gate set."""

import cirq
import numpy as np
import pytest

from cirq_ionq.ionq_native_target_gateset import AriaNativeGateset
from cirq_ionq.ionq_native_target_gateset import ForteNativeGateset

# Tests for transpiling one qubit circuits
qubit1 = cirq.LineQubit.range(2)

@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([1, 0], cirq.Circuit(cirq.I(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.X(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.Y(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.S(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.T(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.X(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.Y(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.Z(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.S(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.T(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.X(qubit1[0]), cirq.X(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.Y(qubit1[0]), cirq.Y(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(qubit1[0]), cirq.Z(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.S(qubit1[0]), cirq.S(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.X(qubit1[0]), cirq.X(qubit1[0]), cirq.X(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.Y(qubit1[0]), cirq.Y(qubit1[0]), cirq.Y(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(qubit1[0]), cirq.Z(qubit1[0]), cirq.Z(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.S(qubit1[0]), cirq.S(qubit1[0]), cirq.S(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.S(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.T(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.H(qubit1[0]), cirq.H(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)


# Tests for transpiling two qubit circuits
qubits2 = cirq.LineQubit.range(2)

@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([1, 0, 0, 0], cirq.Circuit(cirq.SWAP(qubits2[0], qubits2[1]))),
        ([0, 1, 0, 0], cirq.Circuit(cirq.Y(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
        ([0, 1, 0, 0], cirq.Circuit(cirq.X(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
        ([0.5, 0.5, 0, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
        ([0.5, 0.5, 0, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(qubits2[1]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(qubits2[0]), cirq.X(qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(qubits2[0]), cirq.Y(qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(qubits2[0]), cirq.Z(qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.H(qubits2[0]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.X(qubits2[0]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.Y(qubits2[0]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.Z(qubits2[0]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.X(qubits2[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[1]))),
        ([0, 0, 1, 0], cirq.Circuit(cirq.X(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.X(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.Circuit(cirq.X(qubits2[0])))),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]))),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]))),
        ([0, 0, 1, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]))),
        ([0, 0, 1, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.H(qubits2[0]))),
        ([0, 0, 0.5, 0.5], cirq.Circuit(cirq.X(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.H(qubits2[1]))),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.H(qubits2[0]), cirq.H(qubits2[1]))),
        ([0.5, 0.5, 0, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]))),
        ([1, 0, 0.0, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.H(qubits2[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[1], qubits2[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[1], qubits2[0]), cirq.Y(qubits2[1]))),
        ([0, 0.5, 0, 0.5], cirq.Circuit(cirq.H(qubits2[1]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[1], qubits2[0]), cirq.X(qubits2[1]))),
        ([0.5, 0.5, 0, 0], cirq.Circuit(cirq.H(qubits2[1]), cirq.SWAP(qubits2[0], qubits2[1]), cirq.CNOT(qubits2[1], qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1]))),
    ],
)
def test_transpiling_two_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)


# Tests for transpiling three qubit circuits
qubits3 = cirq.LineQubit.range(3)

@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([0.5, 0, 0, 0, 0, 0, 0, 0.5], cirq.Circuit(cirq.H(qubits3[0]), cirq.CNOT(qubits3[0], qubits3[1]), cirq.CNOT(qubits3[1], qubits3[2]))),
        ([0.5, 0, 0, 0, 0, 0, 0, 0.5], cirq.Circuit(cirq.H(qubits3[2]), cirq.CNOT(qubits3[2], qubits3[1]), cirq.CNOT(qubits3[1], qubits3[0]))),
        ([0, 1, 0, 0, 0, 0, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.SWAP(qubits3[0], qubits3[1]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([0, 0, 0, 0, 0, 1, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.SWAP(qubits3[0], qubits3[1]), cirq.CNOT(qubits3[1], qubits3[0]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([0, 0.5, 0, 0, 0, 0.5, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.SWAP(qubits3[0], qubits3[1]), cirq.CNOT(qubits3[1], qubits3[0]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[1]), cirq.CNOT(qubits3[1], qubits3[0]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.CNOT(qubits3[1], qubits3[0]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([0, 0,  0, 0, 0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.CNOT(qubits3[2], qubits3[1]), cirq.SWAP(qubits3[0], qubits3[2]))),
        ([0.25, 0.25, 0.25, 0.25, 0, 0,  0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.CNOT(qubits3[2], qubits3[1]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.Y(qubits3[0]), cirq.Y(qubits3[1]), cirq.Y(qubits3[2]))),
        ([0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.CNOT(qubits3[2], qubits3[1]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.Y(qubits3[1]), cirq.Y(qubits3[2]), cirq.CNOT(qubits3[1], qubits3[0]))),
        ([0, 0.25, 0.25, 0, 0.25, 0, 0, 0.25], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.CNOT(qubits3[1], qubits3[0]), cirq.CNOT(qubits3[2], qubits3[1]))),
        ([0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]), cirq.SWAP(qubits3[1], qubits3[2]))),
        ([1, 0, 0, 0, 0, 0, 0, 0], cirq.Circuit(cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]))),
        ([0, 0, 0, 0, 0, 0, 1, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.X(qubits3[1]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]))),
        ([0, 0, 0, 0, 0, 0, 0.5, 0.5], cirq.Circuit(cirq.X(qubits3[0]), cirq.X(qubits3[1]), cirq.H(qubits3[2]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]))),
        ([0, 0, 0, 0.5, 0, 0, 0, 0.5], cirq.Circuit(cirq.X(qubits3[0]), cirq.X(qubits3[1]), cirq.H(qubits3[2]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]))),
        ([0, 0.5, 0, 0, 0, 0.5, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[2]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]), cirq.SWAP(qubits3[0], qubits3[2]))),
        ([0, 0, 0, 0, 0, 0, 0.5, 0.5], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[2]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]), cirq.CNOT(qubits3[0], qubits3[1]))),
        ([0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0], cirq.Circuit(cirq.X(qubits3[0]), cirq.H(qubits3[1]), cirq.H(qubits3[2]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]), cirq.CNOT(qubits3[1], qubits3[0]))),
    ],
)
def test_transpiling_three_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

