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

import math
import cirq
import numpy as np
import pytest

from cirq_ionq.ionq_native_target_gateset import AriaNativeGateset

one_qubit = cirq.LineQubit.range(2)

@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([1, 0], cirq.Circuit(cirq.I(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.X(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.Y(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.S(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.T(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.X(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.Y(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.H(one_qubit[0]), cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.Z(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.S(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.T(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.X(one_qubit[0]), cirq.X(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.Y(one_qubit[0]), cirq.Y(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(one_qubit[0]), cirq.Z(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.H(one_qubit[0]), cirq.H(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.S(one_qubit[0]), cirq.S(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.X(one_qubit[0]), cirq.X(one_qubit[0]), cirq.X(one_qubit[0]))),
        ([0, 1], cirq.Circuit(cirq.Y(one_qubit[0]), cirq.Y(one_qubit[0]), cirq.Y(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.Z(one_qubit[0]), cirq.Z(one_qubit[0]), cirq.Z(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.S(one_qubit[0]), cirq.S(one_qubit[0]), cirq.S(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.H(one_qubit[0]), cirq.H(one_qubit[0]), cirq.S(one_qubit[0]))),
        ([1, 0], cirq.Circuit(cirq.H(one_qubit[0]), cirq.H(one_qubit[0]), cirq.T(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(one_qubit[0]), cirq.H(one_qubit[0]), cirq.H(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.X_sqrt(one_qubit[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]), cirq.SingleQubitCliffordGate.Y_sqrt(one_qubit[0]))),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)


two_qubits = cirq.LineQubit.range(2)

@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        #([1, 0, 0, 0], cirq.Circuit(cirq.SWAP(two_qubits[0], two_qubits[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(two_qubits[0]), cirq.X(two_qubits[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(two_qubits[0]), cirq.Y(two_qubits[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(two_qubits[0]), cirq.Z(two_qubits[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.H(two_qubits[0]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.X(two_qubits[0]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.Y(two_qubits[0]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.Z(two_qubits[0]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[0]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.Y_sqrt(two_qubits[0]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[0]), cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[1]))),
        ([0, 0.5, 0.5, 0], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.X(two_qubits[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[1]))),
        ([0.25, 0.25, 0.25, 0.25], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CNOT(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.Y_sqrt(two_qubits[1]))),
        ([0, 0, 1, 0], cirq.Circuit(cirq.X(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.X(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]), cirq.Circuit(cirq.X(two_qubits[0])))),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]))),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]))),
        #([0, 0, 1, 0], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]), cirq.SingleQubitCliffordGate.X_sqrt(two_qubits[0]))),
        #([1, 0, 0, 0], cirq.Circuit(cirq.H(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]), cirq.H(two_qubits[0]))),
        ([0, 0, 0.5, 0.5], cirq.Circuit(cirq.X(two_qubits[0]), cirq.CZ(two_qubits[0], two_qubits[1]), cirq.H(two_qubits[1]))),
    ],
)
def test_transpiling_two_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector)**2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)


    


