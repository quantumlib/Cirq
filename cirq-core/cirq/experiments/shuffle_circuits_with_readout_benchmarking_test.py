# Copyright 2019 The Cirq Developers
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

from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.experiments import random_quantum_circuit_generation as rqcg

def test_shuffled_circuits_with_readout_benchmarking_errors_no_noise():
    qubits = cirq.LineQubit.range(5)

    # Generate random input circuits
    input_circuits = []
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.ISWAP**0.5,
        q0=qubits[0],
        q1=qubits[2],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.CNOT**0.5,
        q0=qubits[1],
        q1=qubits[3],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.CNOT**0.5,
        q0=qubits[0],
        q1=qubits[4],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.ISWAP**0.5,
        q0=qubits[2],
        q1=qubits[4],
    )
    for circuit in input_circuits:
        circuit.append(cirq.measure(*qubits, key="m"))

    sampler = cirq.Simulator()
    circuit_repetitions = 1

    measurements, error_rates = cirq.experiments.run_shuffled_with_readout_benchmarking(
        input_circuits, sampler, circuit_repetitions, num_random_bitstrings=100, readout_repetitions=1000
    )
    for measurement in measurements:
        # Five qubits
        assert measurement.shape[1] == 5
    
    for _, (e1, e2)  in error_rates.items():
        assert e1 == 0
        assert e2 == 0

def test_shuffled_circuits_with_readout_benchmarking_errors_with_noise():
    qubits = cirq.LineQubit.range(6)

    # Generate random input circuits
    input_circuits = []
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.ISWAP**0.5,
        q0=qubits[0],
        q1=qubits[1],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.CNOT**0.5,
        q0=qubits[1],
        q1=qubits[3],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.CNOT**0.5,
        q0=qubits[0],
        q1=qubits[4],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.ISWAP**0.5,
        q0=qubits[2],
        q1=qubits[4],
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, 
        two_qubit_gate=cirq.ISWAP**0.5,
        q0=qubits[2],
        q1=qubits[5],
    )
    for circuit in input_circuits:
        circuit.append(cirq.measure(*qubits, key="m"))

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    circuit_repetitions = 1

    measurements, error_rates = cirq.experiments.run_shuffled_with_readout_benchmarking(
        input_circuits, sampler, circuit_repetitions, num_random_bitstrings=100, readout_repetitions=1000
    )
    for measurement in measurements:
        # Five qubits
        assert measurement.shape[1] == 6

    for _, (e1, e2)  in error_rates.items():
        assert 0.08 < e1 < 0.12
        assert 0.18 < e2 < 0.22

def test_empty_input_circuits():
    with pytest.raises(ValueError, match="Input circuits must not be empty."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([], cirq.ZerosSampler(), circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100)

def test_non_circuit_input():
    q = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="Input circuits must be of type cirq.Circuit."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([q], cirq.ZerosSampler(), circuit_repetitions=10, num_random_bitstrings=5,readout_repetitions=100)  # type: ignore

def test_no_measurements():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    with pytest.raises(ValueError, match="Input circuits must have measurements."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([circuit], cirq.ZerosSampler(), circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100)

def test_zero_circuit_repetitions():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide non-zero circuit_repetitions."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([circuit], cirq.ZerosSampler(), circuit_repetitions=0, num_random_bitstrings=5, readout_repetitions=100)

def test_mismatch_circuit_repetitions():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Number of circuit_repetitions must match the number of input circuits."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([circuit], cirq.ZerosSampler(), circuit_repetitions=[10, 20], num_random_bitstrings=5, readout_repetitions=100)

def test_zero_num_random_bitstrings():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide non-zero num_random_bitstrings."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([circuit], cirq.ZerosSampler(), circuit_repetitions=10, num_random_bitstrings=0, readout_repetitions=100)

def test_zero_readout_repetitions():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide non-zero readout_repetitions for readout calibration."):
        cirq.experiments.run_shuffled_with_readout_benchmarking([circuit], cirq.ZerosSampler(), circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=0)

