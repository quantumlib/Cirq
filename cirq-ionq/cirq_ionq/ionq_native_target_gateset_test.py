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

gateset = AriaNativeGateset(atol=1e-8)


# test object representation
def test_AriaNativeGateset_repr():
    gateset = AriaNativeGateset(atol=7)
    assert repr(gateset) == 'cirq_ionq.AriaNativeGateset(atol=7)'


# test object representation
def test_ForteNativeGateset_repr():
    gateset = ForteNativeGateset(atol=7)
    assert repr(gateset) == 'cirq_ionq.ForteNativeGateset(atol=7)'


# test init and atol argument
def test_AriaNativeGateset_init():
    gateset = AriaNativeGateset(atol=7)
    assert gateset.atol == pytest.approx(7)


# test init and atol argument
def test_ForteNativeGateset_init():
    gateset = ForteNativeGateset(atol=7)
    assert gateset.atol == pytest.approx(7)


# test _value_equality_values_ method
def test_AriaNativeGateset__value_equality_values_():
    gateset = AriaNativeGateset(atol=7)
    assert gateset._value_equality_values_() == pytest.approx(7)


# test _value_equality_values_ method
def test_ForteNativeGateset_value_equality_values_():
    gateset = ForteNativeGateset(atol=7)
    assert gateset._value_equality_values_() == pytest.approx(7)


# test _json_dict_ method
def test_AriaNativeGateset__json_dict_():
    gateset = AriaNativeGateset(atol=7)
    assert str(gateset._json_dict_()) == "{'atol': 7}"


# test _json_dict_ method
def test_ForteNativeGateset__json_dict_():
    gateset = ForteNativeGateset(atol=7)
    assert str(gateset._json_dict_()) == "{'atol': 7}"


# test _from_json_dict_ method
def test_AriaNativeGateset__from_json_dict():
    gateset = AriaNativeGateset(atol=7)
    assert repr(gateset._from_json_dict_(7)) == "cirq_ionq.AriaNativeGateset(atol=7)"


# test _from_json_dict_ method
def test_ForteNativeGateset__from_json_dict():
    gateset = ForteNativeGateset(atol=7)
    assert repr(gateset._from_json_dict_(7)) == "cirq_ionq.ForteNativeGateset(atol=7)"


# test _decompose_two_qubit_operation on non unitary argument
def test_AriaNativeGateset_decompose_two_qubit_operation():
    gateset = AriaNativeGateset(atol=7)
    result = gateset._decompose_two_qubit_operation(
        cirq.MeasurementGate(num_qubits=1, key='key'), "blank"
    )
    assert result == NotImplemented


# test _decompose_two_qubit_operation on non unitary argument
def test_ForteNativeGateset_decompose_two_qubit_operation():
    gateset = ForteNativeGateset(atol=7)
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
        (
            [0.5, 0.5],
            cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0])),
        ),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.Y(qubit1[0]))),
        ([0, 1], cirq.Circuit(cirq.H(qubit1[0]), cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.Z(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.S(qubit1[0]))),
        ([0.5, 0.5], cirq.Circuit(cirq.H(qubit1[0]), cirq.T(qubit1[0]))),
        ([1, 0], cirq.Circuit(cirq.X(qubit1[0]), cirq.X(qubit1[0]))),
        (
            [0, 1],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]),
            ),
        ),
        (
            [0, 1],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]),
            ),
        ),
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
        (
            [0.5, 0.5],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubit1[0]),
            ),
        ),
        (
            [0.5, 0.5],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubit1[0]),
            ),
        ),
        (
            [0.387577, 0.612423],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubit1[0]),
                cirq.Circuit(cirq.Ry(rads=2)(qubit1[0])),
                cirq.Circuit(cirq.Rz(rads=1)(qubit1[0])),
            ),
        ),
        (
            [0.551923, 0.448077],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubit1[0]),
                cirq.Circuit(cirq.Ry(rads=2)(qubit1[0])),
                cirq.Circuit(cirq.Rx(rads=3)(qubit1[0])),
            ),
        ),
        (
            [0.257261, 0.742739],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubit1[0]),
                cirq.Circuit(cirq.Rz(rads=2)(qubit1[0])),
                cirq.Circuit(cirq.Rx(rads=3)(qubit1[0])),
            ),
        ),
    ],
)
def test_transpiling_one_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
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
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.SWAP(qubits2[0], qubits2[1])
            ),
        ),
        (
            [0, 0, 0, 1],
            cirq.Circuit(
                cirq.X(qubits2[1]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[0], qubits2[1]),
            ),
        ),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(qubits2[0]), cirq.X(qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(qubits2[0]), cirq.Y(qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(qubits2[0]), cirq.Z(qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.X(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0, 0, 0, 1], cirq.Circuit(cirq.Y(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([1, 0, 0, 0], cirq.Circuit(cirq.Z(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        ([0.5, 0, 0, 0.5], cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]))),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.H(qubits2[0])),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.X(qubits2[0])),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.Y(qubits2[0])),
        ),
        (
            [0.5, 0, 0, 0.5],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.Z(qubits2[0])),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CNOT(qubits2[0], qubits2[1]), cirq.X(qubits2[1])),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[1]),
            ),
        ),
        ([0, 0, 1, 0], cirq.Circuit(cirq.X(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]))),
        (
            [1, 0, 0, 0],
            cirq.Circuit(
                cirq.X(qubits2[0]),
                cirq.CZ(qubits2[0], qubits2[1]),
                cirq.Circuit(cirq.X(qubits2[0])),
            ),
        ),
        ([0.5, 0, 0.5, 0], cirq.Circuit(cirq.H(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]))),
        (
            [0.5, 0, 0.5, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1])
            ),
        ),
        (
            [0, 0, 1, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]),
                cirq.CZ(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[0]),
            ),
        ),
        (
            [0, 0, 1, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]),
                cirq.CZ(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.Y_sqrt(qubits2[0]),
            ),
        ),
        (
            [1, 0, 0, 0],
            cirq.Circuit(cirq.H(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.H(qubits2[0])),
        ),
        (
            [0, 0, 0.5, 0.5],
            cirq.Circuit(cirq.X(qubits2[0]), cirq.CZ(qubits2[0], qubits2[1]), cirq.H(qubits2[1])),
        ),
        (
            [0.5, 0, 0.5, 0],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.H(qubits2[0]),
                cirq.H(qubits2[1]),
            ),
        ),
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]),
            ),
        ),
        (
            [1, 0, 0.0, 0],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[0], qubits2[1]),
                cirq.H(qubits2[1]),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[1], qubits2[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits2[1]),
            ),
        ),
        (
            [0, 0.5, 0.5, 0],
            cirq.Circuit(
                cirq.H(qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[1], qubits2[0]),
                cirq.Y(qubits2[1]),
            ),
        ),
        (
            [0, 0.5, 0, 0.5],
            cirq.Circuit(
                cirq.H(qubits2[1]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[1], qubits2[0]),
                cirq.X(qubits2[1]),
            ),
        ),
        (
            [0.5, 0.5, 0, 0],
            cirq.Circuit(
                cirq.H(qubits2[1]),
                cirq.SWAP(qubits2[0], qubits2[1]),
                cirq.CNOT(qubits2[1], qubits2[0]),
                cirq.SWAP(qubits2[0], qubits2[1]),
            ),
        ),
    ],
)
def test_transpiling_two_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)


# Tests for transpiling three qubit circuits
qubits3 = cirq.LineQubit.range(3)


@pytest.mark.parametrize(
    "ideal_results, circuit",
    [
        (
            [0.5, 0, 0, 0, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.H(qubits3[0]),
                cirq.CNOT(qubits3[0], qubits3[1]),
                cirq.CNOT(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0.5, 0, 0, 0, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.H(qubits3[2]),
                cirq.CNOT(qubits3[2], qubits3[1]),
                cirq.CNOT(qubits3[1], qubits3[0]),
            ),
        ),
        (
            [0, 1, 0, 0, 0, 0, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.SWAP(qubits3[0], qubits3[1]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 1, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.SWAP(qubits3[0], qubits3[1]),
                cirq.CNOT(qubits3[1], qubits3[0]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0.5, 0, 0, 0, 0.5, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.SWAP(qubits3[0], qubits3[1]),
                cirq.CNOT(qubits3[1], qubits3[0]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[1]),
                cirq.CNOT(qubits3[1], qubits3[0]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.CNOT(qubits3[1], qubits3[0]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.CNOT(qubits3[2], qubits3[1]),
                cirq.SWAP(qubits3[0], qubits3[2]),
            ),
        ),
        (
            [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.CNOT(qubits3[2], qubits3[1]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.Y(qubits3[0]),
                cirq.Y(qubits3[1]),
                cirq.Y(qubits3[2]),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.CNOT(qubits3[2], qubits3[1]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.Y(qubits3[1]),
                cirq.Y(qubits3[2]),
                cirq.CNOT(qubits3[1], qubits3[0]),
            ),
        ),
        (
            [0, 0.25, 0.25, 0, 0.25, 0, 0, 0.25],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.CNOT(qubits3[1], qubits3[0]),
                cirq.CNOT(qubits3[2], qubits3[1]),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
                cirq.SWAP(qubits3[1], qubits3[2]),
            ),
        ),
        ([1, 0, 0, 0, 0, 0, 0, 0], cirq.Circuit(cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]))),
        (
            [0, 0, 0, 0, 0, 0, 1, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]), cirq.X(qubits3[1]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2])
            ),
        ),
        (
            [0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0],
            cirq.Circuit(
                cirq.H(qubits3[0]), cirq.H(qubits3[1]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2])
            ),
        ),
        (
            [0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0],
            cirq.Circuit(
                cirq.SingleQubitCliffordGate.X_sqrt(qubits3[0]),
                cirq.SingleQubitCliffordGate.X_sqrt(qubits3[1]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0.224828, 0, 0.545324, 0, 0.067099, 0, 0.162750, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubits3[0]),
                cirq.Rx(rads=2)(qubits3[1]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0.003854, 0, 0.766298, 0, 0.001150, 0, 0.228699, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubits3[0]),
                cirq.Ry(rads=3)(qubits3[1]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0.709106, 0, 0.061046, 0, 0.211630, 0, 0.018219, 0],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubits3[0]),
                cirq.Ry(rads=1)(qubits3[1]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.H(qubits3[1]),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 0.5, 0.5],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.X(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
            ),
        ),
        (
            [0, 0, 0, 0.5, 0, 0, 0, 0.5],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.X(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
            ),
        ),
        (
            [0, 0.5, 0, 0, 0, 0.5, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.SWAP(qubits3[0], qubits3[2]),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 0.5, 0.5],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.CNOT(qubits3[0], qubits3[1]),
            ),
        ),
        (
            [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]),
                cirq.H(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.CNOT(qubits3[1], qubits3[0]),
            ),
        ),
        (
            [0, 0, 0, 0, 0, 0, 1, 0],
            cirq.Circuit(
                cirq.X(qubits3[0]), cirq.X(qubits3[1]), cirq.CCZ(qubits3[0], qubits3[1], qubits3[2])
            ),
        ),
        (
            [0.224828, 0, 0.545324, 0, 0.067099, 0, 0, 0.162750],
            cirq.Circuit(
                cirq.Rx(rads=1)(qubits3[0]),
                cirq.Rx(rads=2)(qubits3[1]),
                cirq.H(qubits3[2]),
                cirq.CCZ(qubits3[0], qubits3[1], qubits3[2]),
                cirq.H(qubits3[2]),
            ),
        ),
    ],
)
def test_transpiling_three_qubit_circuits_to_native_gates(ideal_results, circuit):
    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=AriaNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)

    transpiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ForteNativeGateset())
    simulator = cirq.Simulator()
    result = simulator.simulate(transpiled_circuit)
    probabilities = np.abs(result.final_state_vector) ** 2
    np.testing.assert_allclose(probabilities, ideal_results, atol=1e-3)
