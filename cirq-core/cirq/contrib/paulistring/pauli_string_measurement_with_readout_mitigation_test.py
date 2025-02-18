# Copyright 2025 The Cirq Developers
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
from typing import Dict
import cirq
import numpy as np

from cirq.contrib.paulistring import measure_pauli_strings
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.experiments import SingleQubitReadoutCalibrationResult


def _create_ghz(number_of_qubits: int, qubits: list[cirq.Qid]) -> cirq.Circuit:
    ghz_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, number_of_qubits)],
    )
    return ghz_circuit


def _ideal_expectation_based_on_pauli_string(pauli_string: cirq.PauliString, qubits: list) -> float:
    if pauli_string == cirq.PauliString({q: cirq.X for q in qubits}):
        ideal = 1
    else:
        ideal = 1 if len(qubits) % 2 == 0 else 0
    return ideal


def test_pauli_string_measurement_errors_no_noise():
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""

    qubits = cirq.LineQubit.range(5)
    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))
    sampler = cirq.Simulator()

    circuits_to_pauli: Dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [
        cirq.PauliString({q: cirq.X for q in qubits}),
        cirq.PauliString({q: cirq.Y for q in qubits}),
        cirq.PauliString({q: cirq.Z for q in qubits}),
    ]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, 1000
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)
        assert isinstance(
            circuit_with_pauli_expectations.calibration_result, SingleQubitReadoutCalibrationResult
        )

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            # Since there is no noise, the mitigated and unmitigated expectations should be the same
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                pauli_string_measurement_results.unmitigated_expectation,
            )
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, qubits
                ),
                atol=4 * pauli_string_measurement_results.mitigated_stddev,
            )
        assert circuit_with_pauli_expectations.calibration_result.zero_state_errors == {
            q: 0 for q in qubits
        }
        assert circuit_with_pauli_expectations.calibration_result.one_state_errors == {
            q: 0 for q in qubits
        }


def test_pauli_string_measurement_errors_with_noise():
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.005, seed=1234)

    circuits_to_pauli: Dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [
        cirq.PauliString({q: cirq.Z for q in qubits}),
        cirq.PauliString({q: cirq.X for q in qubits}),
        cirq.PauliString({q: cirq.Y for q in qubits}),
    ]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, np.random.default_rng(), 1000, 1000, 1000
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)
        assert isinstance(
            circuit_with_pauli_expectations.calibration_result, SingleQubitReadoutCalibrationResult
        )

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, qubits
                ),
                atol=4 * pauli_string_measurement_results.mitigated_stddev,
            )
        for error in circuit_with_pauli_expectations.calibration_result.zero_state_errors.values():
            assert 0.08 < error < 0.12
        for error in circuit_with_pauli_expectations.calibration_result.one_state_errors.values():
            assert 0.0045 < error < 0.0055


def test_many_circuits_input_measurement_with_noise():
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string for multiple circuits"""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = [
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
    ]
    qubits_3 = cirq.LineQubit.range(8)

    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))
    circuit_2 = cirq.FrozenCircuit(_create_ghz(5, qubits_2))
    circuit_3 = cirq.FrozenCircuit(_create_ghz(8, qubits_3))

    circuits_to_pauli: Dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit_1] = [
        cirq.PauliString({q: cirq.Z for q in qubits_1}),
        cirq.PauliString({q: cirq.X for q in qubits_1}),
        cirq.PauliString({q: cirq.Y for q in qubits_1}),
    ]
    circuits_to_pauli[circuit_2] = [
        cirq.PauliString({q: cirq.Z for q in qubits_2}),
        cirq.PauliString({q: cirq.X for q in qubits_2}),
        cirq.PauliString({q: cirq.Y for q in qubits_2}),
    ]
    circuits_to_pauli[circuit_3] = [
        cirq.PauliString({q: cirq.Z for q in qubits_3}),
        cirq.PauliString({q: cirq.X for q in qubits_3}),
        cirq.PauliString({q: cirq.Y for q in qubits_3}),
    ]

    sampler = NoisySingleQubitReadoutSampler(p0=0.03, p1=0.005, seed=1234)

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, np.random.default_rng(), 1000, 1000, 1000
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)
        assert isinstance(
            circuit_with_pauli_expectations.calibration_result, SingleQubitReadoutCalibrationResult
        )

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            qubits_to_measure = circuit_with_pauli_expectations.circuit.all_qubits()
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, qubits_to_measure
                ),
                atol=4 * pauli_string_measurement_results.mitigated_stddev,
            )
        for error in circuit_with_pauli_expectations.calibration_result.zero_state_errors.values():
            assert 0.025 < error < 0.035
        for error in circuit_with_pauli_expectations.calibration_result.one_state_errors.values():
            assert 0.0045 < error < 0.0055


def test_allow_measurement_without_readout_mitigation():
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.005, seed=1234)

    circuits_to_pauli: Dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [
        cirq.PauliString({q: cirq.Z for q in qubits}),
        cirq.PauliString({q: cirq.X for q in qubits}),
        cirq.PauliString({q: cirq.Y for q in qubits}),
    ]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, np.random.default_rng(), 1000, 1000, 0
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)
        # Since no readout mitigation string was generated, the calibration
        # result should be None
        assert circuit_with_pauli_expectations.calibration_result is None

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            # Since there's no mitigation, the mitigated and unmitigated expectations
            # should be the same
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                pauli_string_measurement_results.unmitigated_expectation,
            )
        assert circuit_with_pauli_expectations.calibration_result is None
