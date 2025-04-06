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

import numpy as np
import pytest

import cirq
from cirq_ionq.ionq_native_target_gateset import AriaNativeGateset, ForteNativeGateset


# test object representation
def test_AriaNativeGateset_repr():
    gateset = AriaNativeGateset(atol=1e-07)
    assert repr(gateset) == 'cirq_ionq.AriaNativeGateset(atol=1e-07)'


# test object representation
def test_ForteNativeGateset_repr():
    gateset = ForteNativeGateset(atol=1e-07)
    assert repr(gateset) == 'cirq_ionq.ForteNativeGateset(atol=1e-07)'


# test init and atol argument
def test_AriaNativeGateset_init():
    gateset = AriaNativeGateset(atol=1e-07)
    assert gateset.atol == pytest.approx(1e-07)


# test init and atol argument
def test_ForteNativeGateset_init():
    gateset = ForteNativeGateset(atol=1e-07)
    assert gateset.atol == pytest.approx(1e-07)


def test_equality_aria():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(AriaNativeGateset(atol=1e-6))
    eq.add_equality_group(AriaNativeGateset(atol=1e-5))


def test_equality_forte():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(ForteNativeGateset(atol=1e-6))
    eq.add_equality_group(ForteNativeGateset(atol=1e-5))


# test _decompose_two_qubit_operation on non unitary argument
def test_AriaNativeGateset_decompose_two_qubit_operation():
    gateset = AriaNativeGateset(atol=1e-07)
    result = gateset._decompose_two_qubit_operation(
        cirq.MeasurementGate(num_qubits=1, key='key'), "blank"
    )
    assert result == NotImplemented


# test _decompose_two_qubit_operation on non unitary argument
def test_ForteNativeGateset_decompose_two_qubit_operation():
    gateset = ForteNativeGateset(atol=1e-07)
    result = gateset._decompose_two_qubit_operation(
        cirq.MeasurementGate(num_qubits=1, key='key'), "blank"
    )
    assert result == NotImplemented


# test CCZ_gate not working with 2 qubits
def test_CCZ_gate_not_working_with_2_qubits():
    with pytest.raises(Exception) as exc_info:
        gateset = AriaNativeGateset()
        gateset.decompose_all_to_all_connect_ccz_gate(cirq.CCZ, cirq.LineQubit.range(2))
    assert exc_info.value.args[0] == 'Expect 3 qubits for CCZ gate, got 2 qubits.'


# test CCZ_gate not working with 1 qubits
def test_CCZ_gate_not_working_with_1_qubit():
    with pytest.raises(Exception) as exc_info:
        gateset = ForteNativeGateset()
        gateset.decompose_all_to_all_connect_ccz_gate(cirq.CCZ, cirq.LineQubit.range(1))
    assert exc_info.value.args[0] == 'Expect 3 qubits for CCZ gate, got 1 qubits.'


# Tests for transpiling one qubit circuits


@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([1, 0], cirq.Circuit(cirq.I(cirq.q(0)))),
        ([0, 1], cirq.Circuit(cirq.X(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)))),
        ([0, 1], cirq.Circuit(cirq.Y(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.Z(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.S(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.T(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.X(cirq.q(0)))),
        (
            [0.5, 0.5],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0))),
        ),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.Y(cirq.q(0)))),
        ([0, 1], cirq.Circuit(cirq.H(cirq.q(0)), cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.Z(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.S(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.T(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.X(cirq.q(0)), cirq.X(cirq.q(0)))),
        (
            [0, 1],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0, 1],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
            ),
        ),
        ([1, 0], cirq.Circuit(cirq.Y(cirq.q(0)), cirq.Y(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.Z(cirq.q(0)), cirq.Z(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.H(cirq.q(0)), cirq.H(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.S(cirq.q(0)), cirq.S(cirq.q(0)))),
        ([0, 1], cirq.Circuit(cirq.X(cirq.q(0)), cirq.X(cirq.q(0)), cirq.X(cirq.q(0)))),
        ([0, 1], cirq.Circuit(cirq.Y(cirq.q(0)), cirq.Y(cirq.q(0)), cirq.Y(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.Z(cirq.q(0)), cirq.Z(cirq.q(0)), cirq.Z(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.S(cirq.q(0)), cirq.S(cirq.q(0)), cirq.S(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.H(cirq.q(0)), cirq.H(cirq.q(0)), cirq.S(cirq.q(0)))),
        ([1, 0], cirq.Circuit(cirq.H(cirq.q(0)), cirq.H(cirq.q(0)), cirq.T(cirq.q(0)))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.H(cirq.q(0)), cirq.H(cirq.q(0)))),
        (
            [0.5, 0.5],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0.5, 0.5],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0.387577, 0.612423],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Circuit(cirq.Ry(rads=2)(cirq.q(0))),
                cirq.Circuit(cirq.Rz(rads=1)(cirq.q(0))),
            ),
        ),
        (
            [0.551923, 0.448077],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Circuit(cirq.Ry(rads=2)(cirq.q(0))),
                cirq.Circuit(cirq.Rx(rads=3)(cirq.q(0))),
            ),
        ),
        (
            [0.257261, 0.742739],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Circuit(cirq.Rz(rads=2)(cirq.q(0))),
                cirq.Circuit(cirq.Rx(rads=3)(cirq.q(0))),
            ),
        ),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)


# Tests for transpiling two qubit circuits


@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        ([1, 0, 0, 0], cirq.Circuit(cirq.SWAP(cirq.q(0), cirq.q(1)))),
        ([0, 1, 0, 0], cirq.Circuit(cirq.Y(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1)))),
        ([0, 1, 0, 0], cirq.Circuit(cirq.X(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1)))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1)))),
        ([0.5, 0.5, 0, 0], cirq.Circuit(cirq.H(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1)))),
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1))
            ),
        ),
        (
            [0, 0, 0, 1],
            cirq.Circuit(
                cirq.X(cirq.q(1)), cirq.SWAP(cirq.q(0), cirq.q(1)), cirq.CNOT(cirq.q(0), cirq.q(1))
            ),
        ),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(cirq.q(0)), cirq.X(cirq.q(1)))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(cirq.q(0)), cirq.Y(cirq.q(1)))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(cirq.q(0)), cirq.Z(cirq.q(1)))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)))),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.H(cirq.q(0))),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.X(cirq.q(0))),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.Y(cirq.q(0))),
        ),
        (
            [0.5, 0, 0, 0.5],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.Z(cirq.q(0))),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(1)),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.X(cirq.q(1))),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(1)),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(1)),
            ),
        ),
        ([0, 0, 1, 0], cirq.Circuit(cirq.X(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)))),
        (
            [1, 0, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)), cirq.Circuit(cirq.X(cirq.q(0)))
            ),
        ),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.H(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)))),
        (
            [0.5, 0, 0.5, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1))
            ),
        ),
        (
            [0, 0, 1, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.CZ(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
            ),
        ),
        (
            [0, 0, 1, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
                cirq.CZ(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.Y_sqrt(cirq.q(0)),
            ),
        ),
        (
            [1, 0, 0, 0],
            cirq.Circuit(cirq.H(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)), cirq.H(cirq.q(0))),
        ),
        (
            [0, 0, 0.5, 0.5],
            cirq.Circuit(cirq.X(cirq.q(0)), cirq.CZ(cirq.q(0), cirq.q(1)), cirq.H(cirq.q(1))),
        ),
        (
            [0.5, 0, 0.5, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.H(cirq.q(0)),
                cirq.H(cirq.q(1)),
            ),
        ),
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(1)),
            ),
        ),
        (
            [1, 0, 0.0, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
                cirq.H(cirq.q(1)),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(1)),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.Y(cirq.q(1)),
            ),
        ),
        (
            [0, 0.5, 0, 0.5],
            cirq.Circuit(
                cirq.H(cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.X(cirq.q(1)),
            ),
        ),
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.H(cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
            ),
        ),
        (
            [0.224828, 0.545324, 0.162750, 0.067099],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Ry(rads=2)(cirq.q(1)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
            ),
        ),
    ],
)
def test_transpiling_two_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)


# Tests for transpiling three qubit circuits


@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        (
            [0.5, 0, 0, 0, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)), cirq.CNOT(cirq.q(1), cirq.q(2))
            ),
        ),
        (
            [0.5, 0, 0, 0, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.H(cirq.q(2)), cirq.CNOT(cirq.q(2), cirq.q(1)), cirq.CNOT(cirq.q(1), cirq.q(0))
            ),
        ),
        (
            [0, 1, 0, 0, 0, 0, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)), cirq.SWAP(cirq.q(0), cirq.q(1)), cirq.SWAP(cirq.q(1), cirq.q(2))
            ),
        ),
        (
            [0, 0, 0, 0, 0, 1, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SWAP(cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0, 0.5, 0, 0, 0, 0.5, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SWAP(cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(1)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SWAP(cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.SWAP(cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.CNOT(cirq.q(2), cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.CNOT(cirq.q(2), cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.Y(cirq.q(0)),
                cirq.Y(cirq.q(1)),
                cirq.Y(cirq.q(2)),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.CNOT(cirq.q(2), cirq.q(1)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.Y(cirq.q(1)),
                cirq.Y(cirq.q(2)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
            ),
        ),
        (
            [0, 0.25, 0.25, 0, 0.25, 0, 0, 0.25],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
                cirq.CNOT(cirq.q(2), cirq.q(1)),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
                cirq.SWAP(cirq.q(1), cirq.q(2)),
            ),
        ),
        ([1, 0, 0, 0, 0, 0, 0, 0], cirq.Circuit(cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)))),
        (
            [0, 0, 0, 0, 0, 0, 1, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)), cirq.X(cirq.q(1)), cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2))
            ),
        ),
        (
            [0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0],
            cirq.Circuit(
                cirq.H(cirq.q(0)), cirq.H(cirq.q(1)), cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2))
            ),
        ),
        (
            [0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(0)),
                cirq.SingleQubitCliffordGate.X_sqrt(cirq.q(1)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0.224828, 0, 0.545324, 0, 0.067099, 0, 0.162750, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Rx(rads=2)(cirq.q(1)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0.003854, 0, 0.766298, 0, 0.001150, 0, 0.228699, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Ry(rads=3)(cirq.q(1)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0.709106, 0, 0.061046, 0, 0.211630, 0, 0.018219, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Ry(rads=1)(cirq.q(1)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.H(cirq.q(1)),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 0.5, 0.5],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.X(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
            ),
        ),
        (
            [0, 0, 0, 0.5, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.X(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
            ),
        ),
        (
            [0, 0.5, 0, 0, 0, 0.5, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.SWAP(cirq.q(0), cirq.q(2)),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 0.5, 0.5],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.CNOT(cirq.q(0), cirq.q(1)),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)),
                cirq.H(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.CNOT(cirq.q(1), cirq.q(0)),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 1, 0],
            cirq.Circuit(
                cirq.X(cirq.q(0)), cirq.X(cirq.q(1)), cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2))
            ),
        ),
        (
            [0.224828, 0, 0.545324, 0, 0.067099, 0, 0, 0.162750],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Rx(rads=2)(cirq.q(1)),
                cirq.H(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
                cirq.H(cirq.q(2)),
            ),
        ),
        (
            [0.065633, 0.159194, 0.159194, 0.386129, 0.019588, 0.047511, 0.047511, 0.115239],
            cirq.Circuit(
                cirq.Rx(rads=1)(cirq.q(0)),
                cirq.Rx(rads=2)(cirq.q(1)),
                cirq.Ry(rads=2)(cirq.q(2)),
                cirq.CCZ(cirq.q(0), cirq.q(1), cirq.q(2)),
            ),
        ),
    ],
)
def test_transpiling_three_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-5)
